"""Generate metadata.json for fragment_v2 dataset.

Step 1: Use Qwen3-VL to name each image (one word).
Step 2: Use Qwen3-VL to group names into categories (>=10 instances each).
Step 3: Reorganize dirs into image/ and save metadata.json beside it.

Usage:
    python generate_metadata.py
    python generate_metadata.py --model Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

def _get_model_class(model_id: str):
    """Pick the right model class based on model ID."""
    if "Qwen3" in model_id:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration

DATA_ROOT = Path(__file__).resolve().parent / "data" / "fragment_v2"


def load_model(model_id: str, device: str):
    print(f"Loading {model_id} ...")
    cls = _get_model_class(model_id)
    model = cls.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, processor


def name_image(model, processor, image_path: Path, device: str) -> str:
    """Ask Qwen3-VL to name the object in the image with one word."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "What is the single main object in this image? "
                    "Reply with exactly ONE word (a common noun, lowercase). "
                    "No punctuation, no explanation."
                )},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
        )
    # Decode only newly generated tokens
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Extract first word only
    word = response.split()[0].strip(".,;:!?\"'").lower() if response else "unknown"
    return word


def categorize_names(model, processor, names: dict[str, str], device: str) -> dict[str, str]:
    """Ask Qwen3-VL to group object names into categories (>=10 per category)."""
    # Build a summary of all names and their counts
    from collections import Counter
    counts = Counter(names.values())
    name_list = "\n".join(f"  {name}: {count} images" for name, count in counts.most_common())

    prompt = (
        "Below is a list of object names and how many images have that name.\n\n"
        f"{name_list}\n\n"
        "Group these object names into broad categories. Rules:\n"
        "1. Each category must contain at least 10 images total.\n"
        "2. Every object name must be assigned to exactly one category.\n"
        "3. Category names should be short (1-2 words).\n"
        "4. Reply ONLY with a JSON object mapping each object name to its category.\n"
        "   Example: {\"cat\": \"animal\", \"dog\": \"animal\", \"car\": \"vehicle\"}\n"
        "5. No explanation, no markdown, just the JSON."
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=2048, do_sample=False,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Parse JSON from response (may have markdown fences)
    json_str = response
    if "```" in json_str:
        json_str = json_str.split("```")[1]
        if json_str.startswith("json"):
            json_str = json_str[4:]
        json_str = json_str.strip()

    try:
        name_to_category = json.loads(json_str)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse categorization JSON. Raw response:")
        print(f"  {response}")
        print("  Falling back to 'uncategorized' for all.")
        name_to_category = {name: "uncategorized" for name in counts}

    return name_to_category


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else DATA_ROOT
    assert data_root.exists(), f"Data root not found: {data_root}"

    model, processor = load_model(args.model, args.device)

    # Collect all sample dirs
    sample_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name != "image"])
    print(f"\nFound {len(sample_dirs)} samples")

    # Step 1: Name each image
    print("\n=== Step 1: Naming images ===")
    names: dict[str, str] = {}  # dir_name -> object_name
    for i, d in enumerate(sample_dirs):
        img_path = d / "original.png"
        if not img_path.exists():
            print(f"  [{i+1}/{len(sample_dirs)}] {d.name}: SKIP (no original.png)")
            continue
        name = name_image(model, processor, img_path, args.device)
        names[d.name] = name
        if (i + 1) % 10 == 0 or i == 0 or i == len(sample_dirs) - 1:
            print(f"  [{i+1}/{len(sample_dirs)}] {d.name} -> {name}")

    # Step 2: Categorize
    print("\n=== Step 2: Categorizing ===")
    name_to_category = categorize_names(model, processor, names, args.device)

    # Free VRAM
    del model
    torch.cuda.empty_cache()

    # Build per-image metadata
    from collections import Counter
    category_counts = Counter()
    metadata = {"images": []}
    for dir_name, obj_name in sorted(names.items()):
        cat = name_to_category.get(obj_name, "uncategorized")
        category_counts[cat] += 1
        metadata["images"].append({
            "id": dir_name,
            "name": obj_name,
            "category": cat,
        })

    # Print category summary
    print("\n=== Category summary ===")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count} images")

    # Warn about small categories
    small = {cat: count for cat, count in category_counts.items() if count < 10}
    if small:
        print(f"\n  WARNING: {len(small)} categories have <10 images: {small}")
        print("  Consider re-running or manually merging.")

    metadata["categories"] = sorted(category_counts.keys())
    metadata["category_counts"] = dict(category_counts.most_common())

    # Step 3: Reorganize — move sample dirs into image/
    print("\n=== Step 3: Reorganizing ===")
    image_dir = data_root / "image"
    image_dir.mkdir(exist_ok=True)

    for d in sample_dirs:
        dest = image_dir / d.name
        if d == image_dir:
            continue
        if dest.exists():
            print(f"  {d.name} already in image/, skipping move")
            continue
        shutil.move(str(d), str(dest))

    # Save metadata.json beside image/
    meta_path = data_root / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved: {meta_path}")
    print(f"  Images moved to: {image_dir}/")
    print(f"  Total: {len(metadata['images'])} images, {len(metadata['categories'])} categories")


if __name__ == "__main__":
    main()

"""Dataset loaders for fragment completion experiments."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ADE20K_ROOT = (
    "/nfs/turbo/coe-chaijy/jungchun/vault/a-MI/p-visual-grounding/"
    "vit-object-binding/libs/ADE20K/dataset/ADE20K_2021_17_01"
)

DEFAULT_FRAGMENT_V2_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "fragment_v2"
)


def _load_object_info(root: Path) -> dict[int, str]:
    """Parse objectInfo150.csv → {idx: name}."""
    csv_path = root / "objectInfo150.csv"
    idx_to_name = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_name[int(row["Idx"])] = row["Name"].split(";")[0].strip()
    return idx_to_name


def _clean_scene_label(dirname: str) -> str:
    """'atrium__home' → 'atrium home', 'movie_theater__indoor' → 'movie theater'."""
    # Take part before __ if present (drop qualifier like 'indoor', 'home')
    base = dirname.split("__")[0]
    return base.replace("_", " ")


class ADE20KCompletionDataset:
    """Wraps ADE20K validation split (109 images, ~40 scene classes)."""

    def __init__(self, root: str | None = None):
        self.root = Path(root or DEFAULT_ADE20K_ROOT)
        self.idx_to_name = _load_object_info(self.root)

        val_dir = self.root / "images" / "ADE" / "validation"
        self.samples: list[dict] = []
        self._scene_to_id: dict[str, int] = {}

        for jpg_path in sorted(val_dir.rglob("*.jpg")):
            seg_path = jpg_path.with_name(jpg_path.stem + "_seg.png")
            if not seg_path.exists():
                continue

            scene_dir = jpg_path.parent.name
            scene_label = _clean_scene_label(scene_dir)

            if scene_label not in self._scene_to_id:
                self._scene_to_id[scene_label] = len(self._scene_to_id)

            self.samples.append({
                "image_path": jpg_path,
                "seg_path": seg_path,
                "scene_label": scene_label,
                "scene_id": self._scene_to_id[scene_label],
                "image_id": jpg_path.stem,
            })

        self.scene_labels = sorted(self._scene_to_id, key=self._scene_to_id.get)
        self.num_scenes = len(self.scene_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image_pil = Image.open(s["image_path"]).convert("RGB")
        seg_mask = np.array(Image.open(s["seg_path"]))
        # ADE20K seg: R channel encodes class index / 10, G encodes class % 10
        # Combined: class_id = (R/10)*256 + G  — but for 150 classes, simpler:
        # The _seg.png stores class index directly in the red channel for <=150
        # Actually ADE20K stores: R = floor(idx/256), G = idx%256, B = instance
        # For 150 classes: R=0, G=class_idx (1-150), so just use G channel
        if seg_mask.ndim == 3:
            seg_mask = seg_mask[:, :, 0].astype(np.int32) * 256 + seg_mask[:, :, 1].astype(np.int32)
        return {
            "image_pil": image_pil,
            "seg_mask": seg_mask,
            "scene_label": s["scene_label"],
            "scene_id": s["scene_id"],
            "image_id": s["image_id"],
        }


class FragmentV2Dataset:
    """Wraps fragment_v2 dataset (260 images, white-background objects).

    Directory layout (after generate_metadata.py):
      fragment_v2/
        metadata.json       # {images: [{id, name, category}, ...], categories: [...]}
        image/
          001/ original.png, gray.png, lined.png
          002/ ...

    scene_label = category from metadata.json
    scene_id = integer index into the category list
    """

    VALID_TYPES = ("original", "gray", "lined")

    def __init__(self, root: str | Path | None = None, image_type: str = "original"):
        self.root = Path(root) if root else DEFAULT_FRAGMENT_V2_ROOT
        if image_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown image_type '{image_type}'. Choose: {self.VALID_TYPES}"
            )
        self.image_type = image_type

        # Load metadata if available
        meta_path = self.root / "metadata.json"
        image_base = self.root / "image" if (self.root / "image").is_dir() else self.root
        id_to_meta = {}
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self._category_list = meta.get("categories", [])
            self._cat_to_id = {c: i for i, c in enumerate(self._category_list)}
            for entry in meta.get("images", []):
                id_to_meta[entry["id"]] = entry
        else:
            self._category_list = []
            self._cat_to_id = {}

        self.samples: list[dict] = []
        for sample_dir in sorted(image_base.iterdir()):
            if not sample_dir.is_dir():
                continue
            img_path = sample_dir / f"{image_type}.png"
            orig_path = sample_dir / "original.png"
            if not img_path.exists() or not orig_path.exists():
                continue

            dir_name = sample_dir.name
            entry = id_to_meta.get(dir_name, {})
            category = entry.get("category", dir_name)
            obj_name = entry.get("name", dir_name)

            if category not in self._cat_to_id:
                self._cat_to_id[category] = len(self._cat_to_id)
                self._category_list.append(category)

            self.samples.append({
                "image_path": img_path,
                "original_path": orig_path,
                "sample_dir": sample_dir,
                "image_id": dir_name,
                "object_name": obj_name,
                "scene_label": category,
                "scene_id": self._cat_to_id[category],
            })

        self.scene_labels = list(self._category_list)
        self.num_scenes = len(self._category_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image_pil = Image.open(s["image_path"]).convert("RGB")
        # GT foreground mask: always derived from original.png (non-white pixels)
        orig_np = np.array(Image.open(s["original_path"]).convert("RGB"))
        seg_mask = np.where(np.any(orig_np < 250, axis=-1), 1, 0).astype(np.int32)
        return {
            "image_pil": image_pil,
            "seg_mask": seg_mask,
            "scene_label": s["scene_label"],
            "scene_id": s["scene_id"],
            "image_id": s["image_id"],
        }


DEFAULT_COCO_SUBSET_ROOT = (
    Path(__file__).resolve().parent.parent / "data" / "coco_subset"
)


class COCOSubsetDataset:
    """COCO val2017 subset: 20 categories x 50 single-dominant-object images.

    Prepared by ``data/prepare_coco.py``. Stores pre-computed binary masks.

    Directory layout:
      coco_subset/
        metadata.json
        images/    *.jpg
        masks/     *.png (binary foreground)

    Interface matches FragmentV2Dataset: scene_label, scene_id, image_id,
    object_name (= category name, since COCO has no instance names).

    Additionally exposes supercategory info:
      - supercategory_labels: list of supercategory names
      - num_supercategories: count
      - samples[i]["supercategory"] / samples[i]["supercat_id"]
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_COCO_SUBSET_ROOT

        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"COCO subset not prepared. Run:\n"
                f"  uv run python data/prepare_coco.py\n"
                f"Expected: {meta_path}"
            )

        import json
        with open(meta_path) as f:
            meta = json.load(f)

        self._category_list = meta["categories"]
        self._cat_to_id = {c: i for i, c in enumerate(self._category_list)}

        # Supercategory info
        self._supercat_list = meta.get("supercategories", [])
        self._supercat_to_id = {s: i for i, s in enumerate(self._supercat_list)}
        self._cat_to_supercat = meta.get("cat_to_supercat", {})

        self.samples: list[dict] = []
        for entry in meta["images"]:
            img_path = self.root / "images" / entry["file_name"]
            mask_path = self.root / "masks" / entry["file_name"].replace(
                ".jpg", ".png",
            )
            if not img_path.exists():
                continue

            supercat = entry.get("supercategory", "")
            self.samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "image_id": entry["id"],
                "object_name": entry["name"],
                "scene_label": entry["category"],
                "scene_id": self._cat_to_id[entry["category"]],
                "supercategory": supercat,
                "supercat_id": self._supercat_to_id.get(supercat, -1),
            })

        self.scene_labels = list(self._category_list)
        self.num_scenes = len(self._category_list)
        self.supercategory_labels = list(self._supercat_list)
        self.num_supercategories = len(self._supercat_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image_pil = Image.open(s["image_path"]).convert("RGB")
        mask_pil = Image.open(s["mask_path"]).convert("L")
        seg_mask = (np.array(mask_pil) > 127).astype(np.int32)
        return {
            "image_pil": image_pil,
            "seg_mask": seg_mask,
            "scene_label": s["scene_label"],
            "scene_id": s["scene_id"],
            "image_id": s["image_id"],
        }


def get_dataset(name: str, root: str | None = None, image_type: str = "original"):
    """Factory: load dataset by name.

    Args:
        name: "ade20k", "fragment_v2", "coco_subset", or "coco_subset_56"
        root: Optional override for dataset root path.
        image_type: For fragment_v2: "original", "gray", or "lined".
    """
    if name == "ade20k":
        return ADE20KCompletionDataset(root=root)
    elif name == "fragment_v2":
        return FragmentV2Dataset(root=root, image_type=image_type)
    elif name == "coco_subset":
        return COCOSubsetDataset(root=root)
    elif name == "coco_subset_56":
        default_root = (
            Path(__file__).resolve().parent.parent / "data" / "coco_subset_56"
        )
        return COCOSubsetDataset(root=root or default_root)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose: ade20k, fragment_v2, "
            f"coco_subset, coco_subset_56"
        )

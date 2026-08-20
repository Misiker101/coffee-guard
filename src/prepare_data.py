"""
Usage:
    python src/prepare_data.py --raw-dir data/raw/archive --out-dir data/coffee-leaves --val-frac 0.15
"""

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["Healthy", "Miner", "Phoma", "Red Spider Mite", "Rust"]

# Folders that feed into the train/val pool (everything except the
# dedicated test_data folder).
POOL_SOURCES = {
    "main": "Coffee Leaf Diseases/Coffee leaf Diseases",
    "drive": "drive-download-20240530T171920Z-001",
}
TEST_SOURCE = "test_data/content/test_data"


def collect_pool_files(raw_dir: Path):
    """Returns {class_name: [(source_tag, filepath), ...]}"""
    by_class = {c: [] for c in CLASS_NAMES}
    for tag, rel in POOL_SOURCES.items():
        src_dir = raw_dir / rel
        if not src_dir.exists():
            continue
        for class_name in CLASS_NAMES:
            class_dir = src_dir / class_name
            if not class_dir.exists():
                continue
            for f in class_dir.glob("*.jpg"):
                by_class[class_name].append((tag, f))
    return by_class


def collect_test_files(raw_dir: Path):
    by_class = {c: [] for c in CLASS_NAMES}
    test_dir = raw_dir / TEST_SOURCE
    if not test_dir.exists():
        return by_class
    for class_name in CLASS_NAMES:
        class_dir = test_dir / class_name
        if not class_dir.exists():
            continue
        by_class[class_name] = [("test", f) for f in class_dir.glob("*.jpg")]
    return by_class


def copy_split(files_by_class, out_dir: Path, split_name: str):
    total = 0
    for class_name, items in files_by_class.items():
        dest_dir = out_dir / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for tag, src_path in items:
            dest_name = f"{tag}_{src_path.name}"
            shutil.copy2(src_path, dest_dir / dest_name)
            total += 1
    print(f"{split_name}: {total} images across {len(files_by_class)} classes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True,
                         help="Path to the extracted raw archive (contains "
                              "'Coffee Leaf Diseases', 'drive-download-...', 'test_data')")
    parser.add_argument("--out-dir", default="data/coffee-leaves")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    pool = collect_pool_files(raw_dir)
    test = collect_test_files(raw_dir)

    train_split, val_split = {}, {}
    for class_name, items in pool.items():
        items = items[:]
        random.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac)) if items else 0
        val_split[class_name] = items[:n_val]
        train_split[class_name] = items[n_val:]

    copy_split(train_split, out_dir, "train")
    copy_split(val_split, out_dir, "val")
    if any(test.values()):
        copy_split(test, out_dir, "test")
    else:
        print("No test_data folder found in raw-dir; skipping test split "
              "(fine for the sample archive — re-run against the full dataset later).")

    print(f"\nDone. Clean dataset written to: {out_dir.resolve()}")
    print("Point src/train.py --data-dir at this folder.")


if __name__ == "__main__":
    main()

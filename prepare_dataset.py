import argparse
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "road_damage_data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Road Damage Dataset (2-class or 3-class).")
    parser.add_argument(
        "--classes",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of classes to output: 2 (pothole, manhole) or 3 (pothole, crack, manhole).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode_3class = args.classes == 3

    out_folder_name = "road_damage_3class" if mode_3class else "road_damage_2class"
    output_dir = PROJECT_DIR / out_folder_name
    yaml_filename = "data_3class.yaml" if mode_3class else "data_2class.yaml"
    yaml_path = PROJECT_DIR / yaml_filename

    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not find source dataset at: {SOURCE_DIR}\n"
            "Please ensure road_damage_data/ exists with images/ and labels/ subdirectories."
        )

    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists. Cleaning before write...")
        shutil.rmtree(output_dir)

    print(f"Preparing {args.classes}-Class Dataset...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {output_dir}")

    # Original IDs: 0=pothole, 1=crack, 2=manhole
    if mode_3class:
        keep_and_remap = {0: 0, 1: 1, 2: 2}
    else:
        keep_and_remap = {0: 0, 2: 1}  # Drop crack (1), remap manhole (2 -> 1)

    totals = {
        split: {"copied": 0, "skipped": 0, "empty_labels": 0}
        for split in SPLITS
    }

    for split in SPLITS:
        images_dir = SOURCE_DIR / "images" / split
        labels_dir = SOURCE_DIR / "labels" / split

        if not images_dir.is_dir() or not labels_dir.is_dir():
            print(f"Notice: Split '{split}' missing in source dataset, skipping.")
            continue

        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(images_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue

            lines = [line.strip() for line in lbl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept_lines = []
            original_classes = set()

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(parts[0])
                    coords = [float(v) for v in parts[1:]]
                except ValueError:
                    continue

                original_classes.add(cid)
                if cid in keep_and_remap:
                    new_cid = keep_and_remap[cid]
                    kept_lines.append(f"{new_cid} " + " ".join(parts[1:]))

            if not mode_3class and original_classes == {1}:
                totals[split]["skipped"] += 1
                continue

            shutil.copy2(img_path, out_images / img_path.name)
            (out_labels / lbl_path.name).write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
            totals[split]["copied"] += 1
            if not kept_lines:
                totals[split]["empty_labels"] += 1

    # Write YAML
    if mode_3class:
        names_yaml = "  0: pothole\n  1: crack\n  2: manhole\n"
    else:
        names_yaml = "  0: pothole\n  1: manhole\n"

    yaml_text = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
{names_yaml}"""
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("\nDataset preparation completed successfully!")
    print(f"Output directory : {output_dir}")
    print(f"Config YAML file : {yaml_path}")
    print("\nSplit statistics:")
    for split, counts in totals.items():
        print(f"  [{split}] Copied: {counts['copied']} | Skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()

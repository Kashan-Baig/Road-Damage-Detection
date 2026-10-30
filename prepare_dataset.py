from pathlib import Path
import shutil

# Expected project structure:
# RoadDamageDetection/
#   road_damage_data/images/{train,val,test}
#   road_damage_data/labels/{train,val,test}

PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "road_damage_data"
OUTPUT_DIR = PROJECT_DIR / "road_damage_2class"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

# Original IDs: 0=pothole, 1=crack, 2=manhole
# New IDs:      0=pothole, 1=manhole
KEEP_AND_REMAP = {0: 0, 2: 1}
CRACK_ID = 1


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Could not find {SOURCE_DIR}. Run prepare_dataset.py first."
        )

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Rename/delete that folder first, or change OUTPUT_DIR in this script."
        )

    totals = {split: {"copied": 0, "skipped_crack_only": 0, "empty_labels": 0}
              for split in SPLITS}

    for split in SPLITS:
        images_dir = SOURCE_DIR / "images" / split
        labels_dir = SOURCE_DIR / "labels" / split

        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected both directories for split '{split}':\n"
                f"{images_dir}\n{labels_dir}"
            )

        out_images = OUTPUT_DIR / "images" / split
        out_labels = OUTPUT_DIR / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing label file for image:\n{image_path}\nExpected:\n{label_path}"
                )

            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
            kept_lines = []
            original_classes = set()

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label ({label_path}, line {line_number}): {line}")
                try:
                    class_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric YOLO label ({label_path}, line {line_number}): {line}") from exc

                if class_id not in (0, 1, 2):
                    raise ValueError(f"Unexpected class ID {class_id} in {label_path}")
                if not all(0 <= value <= 1 for value in coords):
                    raise ValueError(f"Coordinates outside [0, 1] in {label_path}: {line}")

                original_classes.add(class_id)
                if class_id in KEEP_AND_REMAP:
                    new_class_id = KEEP_AND_REMAP[class_id]
                    kept_lines.append(" ".join([str(new_class_id), *parts[1:]]))
                # class 1 (crack) is intentionally removed

            # If an image contains cracks only, omit it rather than teaching
            # the model that visible crack damage is a background example.
            if original_classes and original_classes.issubset({CRACK_ID}):
                totals[split]["skipped_crack_only"] += 1
                continue

            shutil.copy2(image_path, out_images / image_path.name)
            (out_labels / label_path.name).write_text(
                "\n".join(kept_lines) + ("\n" if kept_lines else ""),
                encoding="utf-8",
            )
            totals[split]["copied"] += 1
            if not kept_lines:
                totals[split]["empty_labels"] += 1

    yaml_text = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
  1: manhole
"""
    yaml_path = PROJECT_DIR / "data_2class.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("Created two-class dataset successfully.")
    print(f"Dataset folder: {OUTPUT_DIR}")
    print(f"Dataset YAML:   {yaml_path}")
    print("\nCounts:")
    for split, counts in totals.items():
        print(
            f"{split}: copied={counts['copied']}, "
            f"crack-only images skipped={counts['skipped_crack_only']}, "
            f"images with no remaining target boxes={counts['empty_labels']}"
        )
    print("\nClass mapping: old pothole 0 -> new pothole 0; old manhole 2 -> new manhole 1")


if __name__ == "__main__":
    main()

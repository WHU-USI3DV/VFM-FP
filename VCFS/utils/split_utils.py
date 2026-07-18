"""Dataset split helpers for VCFS training."""

from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def read_split_names(path):
    path = Path(path)
    if not path.exists():
        return []

    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            names.append(parts[0])
    return names


def write_split_names(path, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def paired_stems(dataset_path):
    dataset_path = Path(dataset_path)
    image_dir = dataset_path / "JPEGImages"
    label_dir = dataset_path / "SegmentationClass"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Expected JPEGImages and SegmentationClass under {dataset_path}")

    image_stems = set()
    for extension in IMAGE_EXTENSIONS:
        image_stems.update(path.stem for path in image_dir.glob(f"*{extension}"))

    label_stems = {path.stem for path in label_dir.glob("*.png")}
    return sorted(image_stems & label_stems)


def create_augmented_train_split(
    dataset_path,
    output_name="train_1601.txt",
    base_train_name="train.txt",
    val_name="val.txt",
    test_name="test.txt",
    generated_prefix="syn_",
):
    """Create the train split used by the SDA-augmented setting."""

    dataset_path = Path(dataset_path)
    split_dir = dataset_path / "txt"
    paired = set(paired_stems(dataset_path))

    base_train = [name for name in read_split_names(split_dir / base_train_name) if name in paired]
    excluded = set(read_split_names(split_dir / val_name)) | set(read_split_names(split_dir / test_name))
    generated = sorted(
        name
        for name in paired
        if name.startswith(generated_prefix) and name not in excluded and name not in base_train
    )

    names = base_train + generated if base_train else sorted(name for name in paired if name not in excluded)
    if not names:
        raise RuntimeError(f"No paired training samples found under {dataset_path}")

    output_path = split_dir / output_name
    write_split_names(output_path, names)
    return str(output_path), len(names)


def load_split_lines(dataset_path, preferred_name, fallback_name=None, auto_create_train=False):
    split_dir = Path(dataset_path) / "txt"
    preferred_path = split_dir / preferred_name

    if preferred_path.exists() and preferred_path.stat().st_size > 0:
        return preferred_path.read_text(encoding="utf-8").splitlines(True), str(preferred_path)

    if auto_create_train:
        created_path, count = create_augmented_train_split(
            dataset_path,
            output_name=preferred_name,
            base_train_name=fallback_name or "train.txt",
        )
        print(f"Created {created_path} with {count} samples.")
        return Path(created_path).read_text(encoding="utf-8").splitlines(True), created_path

    if fallback_name:
        fallback_path = split_dir / fallback_name
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            print(f"Using fallback split {fallback_path}.")
            return fallback_path.read_text(encoding="utf-8").splitlines(True), str(fallback_path)

    raise FileNotFoundError(f"Could not find split {preferred_name} under {split_dir}")

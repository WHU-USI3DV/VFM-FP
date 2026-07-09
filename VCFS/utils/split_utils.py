"""Dataset split helpers for VCFS training.

The accepted experiments used train_1601.txt after SDA expansion.  Public
examples may only contain train.txt, so these helpers create a compatible
train_1601.txt from available paired images and labels when needed.
"""

from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _read_names(path):
    path = Path(path)
    if not path.exists():
        return []
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip().split()
        if name:
            names.append(name[0])
    return names


def _write_names(path, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def _paired_stems(dataset_path):
    dataset_path = Path(dataset_path)
    image_dir = dataset_path / "JPEGImages"
    label_dir = dataset_path / "SegmentationClass"

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Expected JPEGImages and SegmentationClass under {}".format(dataset_path)
        )

    image_stems = set()
    for ext in IMAGE_EXTENSIONS:
        image_stems.update(path.stem for path in image_dir.glob("*" + ext))

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
    """Create a train split compatible with the SDA-augmented setting.

    The generated split keeps the base train.txt order, then appends paired
    generated samples whose names start with generated_prefix. Validation and
    test names are excluded when those files exist.
    """

    dataset_path = Path(dataset_path)
    split_dir = dataset_path / "txt"
    paired = set(_paired_stems(dataset_path))

    base_train = [name for name in _read_names(split_dir / base_train_name) if name in paired]
    val_names = set(_read_names(split_dir / val_name))
    test_names = set(_read_names(split_dir / test_name))
    excluded = val_names | test_names

    generated = sorted(
        name for name in paired
        if name.startswith(generated_prefix) and name not in excluded and name not in base_train
    )

    if base_train:
        names = base_train + generated
    else:
        names = sorted(name for name in paired if name not in excluded)

    if not names:
        raise RuntimeError("No paired training samples found under {}".format(dataset_path))

    output_path = split_dir / output_name
    _write_names(output_path, names)
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
        print("Created {} with {} samples.".format(created_path, count))
        return Path(created_path).read_text(encoding="utf-8").splitlines(True), created_path

    if fallback_name is not None:
        fallback_path = split_dir / fallback_name
        if fallback_path.exists() and fallback_path.stat().st_size > 0:
            print("Using fallback split {}.".format(fallback_path))
            return fallback_path.read_text(encoding="utf-8").splitlines(True), str(fallback_path)

    raise FileNotFoundError(
        "Could not find split {} under {}".format(preferred_name, split_dir)
    )

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from src.dataset import get_dataset
from src.preprocessing import detect_metadata_start

dataset = get_dataset()
rng = random.Random(42)

output_dir = Path("outputs/crop_audit")
output_dir.mkdir(parents=True, exist_ok=True)

for class_idx, class_name in enumerate(dataset.classes):

    paths = [
        path for path, label in dataset.samples
        if label == class_idx
    ]

    selected = rng.sample(paths, min(10, len(paths)))

    sheet = Image.new(
        "RGB",
        (660, len(selected) * 220 + 35),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    draw.text((10, 10), class_name, fill="black")

    detected = 0

    for i, path in enumerate(selected):
        with Image.open(path) as img:
            original = img.convert("RGB")

        crop_y = detect_metadata_start(original)

        if crop_y is not None:
            detected += 1
            cropped = original.crop(
                (0, 0, original.width, crop_y)
            )
        else:
            cropped = original

        left = ImageOps.contain(original, (310, 185))
        right = ImageOps.contain(cropped, (310, 185))

        y = 35 + i * 220

        sheet.paste(left, (5, y))
        sheet.paste(right, (335, y))

        status = "detected" if crop_y is not None else "NOT DETECTED"
        draw.text((335, y + 188), status, fill="black")

    sheet.save(output_dir / f"{class_idx:02d}_{class_name}.jpg")

    print(f"{class_name}: {detected}/{len(selected)} detected")
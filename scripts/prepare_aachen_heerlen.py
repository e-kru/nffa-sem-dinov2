"""Convert Aachen-Heerlen polygon annotations to segmentation masks."""

import ast
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.config import AACHEN_ROOT


ROOT = AACHEN_ROOT
RAW = ROOT / "raw"
IMAGES = ROOT / "images"
MASKS = ROOT / "masks"

IMAGES.mkdir(parents=True, exist_ok=True)
MASKS.mkdir(parents=True, exist_ok=True)

# 1. Polygonannotationen laden
annotations = defaultdict(list)

csv_path = RAW / "nature_scidata_heerlen_aachen_steel_annotations_polygon.csv"

with csv_path.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        polygon = ast.literal_eval(row["polygon"])
        annotations[row["image_url"]].append(polygon)

print("Bilder mit Annotationen:", len(annotations))
print("Polygone:", sum(map(len, annotations.values())))

# 2. Bilder lesen und Masken erzeugen
positive_fractions = []

with zipfile.ZipFile(RAW / "PNG.zip") as archive:
    available = {
        Path(name).name: name
        for name in archive.namelist()
        if name.lower().endswith(".png")
    }

    assert len(available) == len(annotations), (
        "Bildanzahl und Anzahl annotierter Bilder stimmen nicht überein"
    )
    assert set(available) == set(annotations), (
        "Bildnamen und Annotationsnamen stimmen nicht überein"
    )

    for index, name in enumerate(sorted(annotations), start=1):
        image_bytes = archive.read(available[name])

        with Image.open(io.BytesIO(image_bytes)) as image:
            size = image.size

        # Pixelwerte: 0 = Hintergrund, 1 = MA-Insel
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)

        for polygon in annotations[name]:
            if len(polygon) < 3:
                raise ValueError(f"Ungültiges Polygon in {name}")

            # Expertenpolygone dürfen bis zu 2 Pixel über den Bildrand reichen.
            # Die ursprünglichen Koordinaten bleiben unverändert.
            tolerance = 2

            for x, y in polygon:
                if not (
                    -tolerance <= x <= size[0] - 1 + tolerance
                    and -tolerance <= y <= size[1] - 1 + tolerance
                ):
                    raise ValueError(
                        f"Polygon deutlich außerhalb: {name}, ({x}, {y})"
                    )

            draw.polygon(polygon, fill=1)

            draw.polygon(polygon, fill=1)

        # Originalbild unverändert speichern
        (IMAGES / name).write_bytes(image_bytes)

        # Trainingsmaske mit Klassenindizes speichern
        mask.save(MASKS / name)

        array = np.asarray(mask)
        assert set(np.unique(array)).issubset({0, 1})
        assert np.any(array == 1), f"Leere Maske: {name}"

        positive_fractions.append(float((array == 1).mean()))

        if index % 200 == 0:
            print(f"{index}/{len(annotations)} verarbeitet")

print("\nFERTIG")
print("Bilder:", len(list(IMAGES.glob("*.png"))))
print("Masken:", len(list(MASKS.glob("*.png"))))
print("Mittlerer MA-Flächenanteil:",
      round(100 * np.mean(positive_fractions), 2), "%")
print("Bilder unter:", IMAGES)
print("Masken unter:", MASKS)

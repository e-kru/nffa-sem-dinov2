
import numpy as np
from PIL import Image


def detect_metadata_start(image):
    """
    Detect the upper boundary of a white SEM metadata strip.
    Returns the crop position or None.
    """

    width, height = image.size

    if height < 80:
        return None

    # Reduce width only, preserve vertical resolution
    small = image.convert("RGB").resize(
        (min(width, 512), height),
        Image.Resampling.BILINEAR
    )

    pixels = np.asarray(small)

    # Fraction of nearly white pixels in each row
    white_fraction = (
        np.min(pixels, axis=2) >= 230
    ).mean(axis=1)

    # Short windows to detect thin metadata strips
    window = max(4, round(0.007 * height))
    band_height = max(8, round(0.014 * height))

    # Search in the bottom 25%
    for y in range(
        int(0.75 * height),
        height - band_height
    ):
        above = white_fraction[y - window:y].mean()
        below = white_fraction[y:y + window].mean()

        band = white_fraction[y:y + band_height]

        # Darker image above, bright strip below
        if above > 0.45:
            continue

        if below < 0.75:
            continue

        # Strong transition
        if below - above < 0.50:
            continue

        # Bright region must continue for several rows
        if (band >= 0.65).mean() < 0.65:
            continue

        return max(1, y - 2)

    return None


def crop_metadata_strip(image):
    """Remove the metadata strip if detected."""

    crop_y = detect_metadata_start(image)

    if crop_y is None:
        return image

    return image.crop((0, 0, image.width, crop_y))

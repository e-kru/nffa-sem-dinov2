import numpy as np
from PIL import Image


def detect_metadata_start(image):
    """
    Finds the upper edge of a white SEM metadata strip.
    Returns the crop position or None.
    """

    width, height = image.size

    # Downsample for fast detection, independent of resolution
    small_height = 192
    small_width = max(64, round(width * small_height / height))

    small = image.convert("RGB").resize(
        (small_width, small_height),
        Image.Resampling.BILINEAR
    )

    pixels = np.asarray(small)

    # Fraction of nearly white pixels in each row
    white_fraction = np.all(pixels >= 235, axis=2).mean(axis=1)

    min_band = max(8, round(0.055 * small_height))

    # Search only in the lower part of the image
    for y in range(
        int(0.55 * small_height),
        int(0.90 * small_height)
    ):
        # White horizontal border
        if white_fraction[y] < 0.90:
            continue

        # Region above should be substantially darker
        if white_fraction[y-5:y].mean() > 0.45:
            continue

        # Check for an extended bright metadata strip
        band = white_fraction[y:y + min_band]

        if band.mean() < 0.55:
            continue

        if (band > 0.30).mean() < 0.70:
            continue

        # Convert to original image coordinates
        crop_y = round(y * height / small_height)

        # Small safety margin above the detected boundary
        crop_y -= max(1, round(0.006 * height))

        return max(1, crop_y)

    return None


def crop_metadata_strip(image):
    """Remove the metadata strip if one is detected."""

    crop_y = detect_metadata_start(image)

    if crop_y is None:
        return image

    return image.crop((0, 0, image.width, crop_y))
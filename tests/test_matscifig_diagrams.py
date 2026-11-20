import cv2
import numpy as np
from PIL import Image

from scripts.audit_matscifig_diagrams import (
    axis_features,
    diagram_score,
    measure,
    texture_features,
)


def test_white_thresholds_use_all_native_rgb_channels(tmp_path):
    path = tmp_path / 'sample.png'
    image = np.full((52, 400, 3), 255, dtype=np.uint8)
    image[:, :100, 0] = 224
    image[:, 100:200, 0] = 225
    image[:, 200:300, 0] = 235
    Image.fromarray(image).save(path)
    record = dict(image_id='sample', path=str(path), native_pixels=20800,
                  strong_color_pixels=0, near_white_pixels=5200)

    result = measure(record)

    assert not result['error']
    assert [result[f'white{threshold}_pixels'] for threshold in (225, 235, 245)] == [
        15600, 10400, 5200,
    ]
    assert measure({**record, 'near_white_pixels': 0})['error']


def test_hough_axes_require_a_pair_and_reject_a_closed_frame():
    white = np.full((320, 384), 255, dtype=np.uint8)
    plot = white.copy()
    cv2.line(plot, (55, 40), (55, 265), 0, 2)
    cv2.line(plot, (55, 265), (349, 265), 0, 2)
    frame = white.copy()
    cv2.rectangle(frame, (55, 40), (349, 265), 0, 2)
    scale_bar = white.copy()
    cv2.line(scale_bar, (60, 275), (190, 275), 0, 3)

    assert axis_features(plot)['axis_evidence'] > 0.9
    assert axis_features(frame)['frame_pairs_rejected'] > 0
    assert axis_features(frame)['axis_pair'] == 0
    bar = axis_features(scale_bar)
    assert bar['hough_horizontal'] > 0
    assert bar['axis_pair'] == 0


def test_microscopy_texture_overrides_bright_background_and_axis_evidence():
    gray = np.full((320, 384), 255, dtype=np.uint8)
    gray[8:146, 8:182] = np.random.default_rng(42).integers(40, 210, (138, 174), dtype=np.uint8)
    features = texture_features(gray)
    assert features['large_flat_fraction'] > 0.7
    assert features['dense_texture_fraction'] >= 0.16

    row = dict(features, native_pixels=gray.size, axis_evidence=1)
    row.update({f'white{level}_pixels': int((gray >= level).sum()) for level in (225, 235, 245)})

    score, protected = diagram_score(row)
    assert protected
    assert score <= 49

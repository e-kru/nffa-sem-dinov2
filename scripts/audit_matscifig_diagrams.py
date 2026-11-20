"""CPU-only MatSciFig diagram triage. Writes an audit, never a training manifest."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.audit_matscifig_sem_clip import DATASET, TRAINING_EXPORT, load_image, write_csv
from scripts.build_sem_posttraining_manifest import metadata_sha, read_json, rows


PIXEL_AUDIT = Path('outputs/dataset_audits/matscifig_color_white_count_v0')
THRESHOLDS = (40, 50, 55, 60, 65, 70, 75, 80, 85, 90)
REVIEW_THRESHOLD = 70
MAX_SIDE = 384
WHITE_LEVELS = (225, 235, 245)
EXAMPLES = 100
FEATURE_FIELDS = (
    'white225_pixels', 'white235_pixels', 'white245_pixels',
    'largest_flat_fraction', 'large_flat_fraction', 'local_gray_std_mean',
    'strong_texture_fraction', 'dense_texture_fraction', 'hough_horizontal',
    'hough_vertical', 'frame_pairs_rejected', 'axis_pair', 'axis_field_white235',
    'axis_evidence',
)


def initialize_worker():
    cv2.setNumThreads(1)


def ramp(value, lo, hi):
    return min(1., max(0., (value - lo) / (hi - lo)))


def axis_features(gray):
    """Find interior bottom-left L pairs, excluding closed rectangular borders."""
    height, width = gray.shape
    short = min(height, width)
    edges = cv2.Canny(gray, 45, 115)
    raw_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(16, round(.10 * short)),
                               minLineLength=max(18, round(.20 * short)),
                               maxLineGap=max(2, round(.02 * short)))
    horizontal, vertical = [], []
    for x0, y0, x1, y1 in (() if raw_lines is None else raw_lines.reshape(-1, 4)):
        dx, dy = abs(int(x1) - int(x0)), abs(int(y1) - int(y0))
        if dx >= .29 * width and dy <= max(1, .035 * dx):
            horizontal.append((min(x0, x1), max(x0, x1), round((int(y0)+int(y1))/2)))
        if dy >= .29 * height and dx <= max(1, .035 * dy):
            vertical.append((min(y0, y1), max(y0, y1), round((int(x0)+int(x1))/2)))
    horizontal = sorted(set(horizontal), key=lambda line: (line[0]-line[1], line))[:24]
    vertical = sorted(set(vertical), key=lambda line: (line[0]-line[1], line))[:24]
    corner_tol = max(4, round(.045 * short))
    opposite_tol = max(5, round(.065 * short))
    frame_pairs = 0
    best = (0., 0.)
    has_pair = False
    for left, right, y in horizontal:
        if not .47*height <= y <= .94*height:
            continue
        for top, bottom, x in vertical:
            if not .06*width <= x <= .53*width:
                continue
            if abs(left-x) > corner_tol or abs(bottom-y) > corner_tol:
                continue
            # Look for both opposite sides at the endpoints of the L.
            top_side = any(abs(other_y-top) <= opposite_tol and abs(lo-x) <= opposite_tol
                           and abs(hi-right) <= opposite_tol for lo, hi, other_y in horizontal)
            right_side = any(abs(other_x-right) <= opposite_tol and abs(lo-top) <= opposite_tol
                             and abs(hi-y) <= opposite_tol for lo, hi, other_x in vertical)
            if top_side and right_side:
                frame_pairs += 1
                continue
            has_pair = True
            field = gray[top+2:y-2, x+2:right-2]
            if field.size == 0:
                continue
            light = float(np.mean(field >= 235))
            span = min(1., (right-x)/(.60*width), (y-top)/(.60*height))
            support = span * ramp(light, .25, .80)
            best = max(best, (support, light))
    return dict(hough_horizontal=len(horizontal), hough_vertical=len(vertical),
                frame_pairs_rejected=frame_pairs, axis_pair=int(has_pair),
                axis_field_white235=best[1], axis_evidence=best[0])


def texture_features(gray):
    gray_float = gray.astype(np.float32)
    mean = cv2.boxFilter(gray_float, -1, (9, 9))
    mean_square = cv2.sqrBoxFilter(gray_float, cv2.CV_32F, (9, 9))
    variance = np.maximum(mean_square - mean * mean, 0)
    flat = cv2.morphologyEx((variance <= 9).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), dtype=np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    areas = components[1:, cv2.CC_STAT_AREA]
    textured = (variance >= 64).astype(np.float32)
    # Dense texture across a grid cell is stronger evidence than a thin line,
    # and a nearly white grid cell containing text should not count as SEM.
    density = cv2.resize(textured, (8, 8), interpolation=cv2.INTER_AREA)
    light = cv2.resize((gray >= 235).astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
    return dict(largest_flat_fraction=int(areas.max(initial=0))/gray.size,
                large_flat_fraction=int(areas[areas >= .02*gray.size].sum())/gray.size,
                local_gray_std_mean=float(np.sqrt(variance).mean()),
                strong_texture_fraction=float(textured.mean()),
                dense_texture_fraction=float(np.mean((density >= .55) & (light <= .40))))


def measure(row):
    keys = ('image_id', 'path', 'native_pixels', 'strong_color_pixels')
    result = {key: row[key] for key in keys}
    try:
        image = cv2.imread(row['path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError('JPEG decode failed')
        height, width = image.shape[:2]
        if width*height != int(row['native_pixels']):
            raise ValueError('Native pixel count changed since color audit')
        first, second, third = cv2.split(image)
        lowest = cv2.min(cv2.min(first, second), third)
        white = {level: cv2.countNonZero(cv2.compare(lowest, level, cv2.CMP_GE))
                 for level in WHITE_LEVELS}
        if white[245] != int(row['near_white_pixels']):
            raise ValueError('Native white245 count changed since color audit')
        del first, second, third, lowest
        scale = min(1., MAX_SIDE / max(height, width))
        if scale < 1:
            image = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result.update({f'white{level}_pixels': count for level, count in white.items()})
        result.update(texture_features(gray))
        result.update(axis_features(gray))
        result['error'] = ''
    except (ValueError, OSError, cv2.error) as error:
        result = result | dict.fromkeys(FEATURE_FIELDS, '') | {'error': str(error)}
    return result


def extract(output, pixel_audit, workers):
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'features.csv'
    if target.exists():
        raise FileExistsError(target)
    pixel_counts = pixel_audit / 'pixel_counts.csv'
    provenance = read_json(pixel_audit / 'summary.json')
    export_hash = metadata_sha(TRAINING_EXPORT)
    if export_hash != provenance['training_export_sha256']:
        raise ValueError('V0 export differs from the previous pixel audit')
    expected = {row['image_id']: row['path'] for row in rows(TRAINING_EXPORT)
                if row['source'] == 'matscifig_sem' and row['included'] == '1'}
    records = list(rows(pixel_counts))
    if (len(records) != len(expected) or {row['image_id']: row['path'] for row in records} != expected
        or len(set(expected.values())) != len(expected)):
        raise ValueError('The pixel audit must match every active MatSciFig row exactly')
    for row in records:
        path = Path(row['path'])
        if row['error'] or not path.is_relative_to(DATASET / 'images') or '..' in path.parts:
            raise ValueError(f'Unexpected pixel-audit row {path}')

    start = time.monotonic()
    temporary = target.with_suffix('.partial.csv')
    with temporary.open('x', newline='') as stream, ProcessPoolExecutor(
        max_workers=workers, initializer=initialize_worker
    ) as pool:
        fields = ('image_id', 'path', 'native_pixels', 'strong_color_pixels', *FEATURE_FIELDS, 'error')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        errors = 0
        for i, result in enumerate(pool.map(measure, records, chunksize=12), 1):
            writer.writerow(result)
            errors += bool(result['error'])
            if i % 5000 == 0 or i == len(records):
                stream.flush()
                print(f'{i}/{len(records)} measured, {errors} errors, {time.monotonic()-start:.0f} s', flush=True)
    if metadata_sha(TRAINING_EXPORT) != export_hash:
        raise ValueError('The active training export changed during the audit')
    temporary.rename(target)
    write_csv(output / 'provenance.csv', [dict(
        image_count=len(records), decode_or_audit_errors=errors, training_export_sha256=export_hash,
        input_pixel_counts_sha256=metadata_sha(pixel_counts), features_sha256=metadata_sha(target),
        measurement_source_sha256=metadata_sha(Path(__file__)), opencv=cv2.__version__, analysis_max_side=MAX_SIDE,
        seconds=round(time.monotonic()-start, 1))])


def diagram_score(row):
    pixels = int(row['native_pixels'])
    white225, white235, white245 = (int(row[f'white{level}_pixels'])/pixels for level in WHITE_LEVELS)
    light = ramp(.50*white225 + .30*white235 + .20*white245, .20, .80)
    flat = ramp(float(row['large_flat_fraction']), .30, .90)
    low_texture = 1-ramp(float(row['strong_texture_fraction']), .06, .28)
    axes = float(row['axis_evidence'])
    density = float(row['dense_texture_fraction'])
    penalty = .65*ramp(density, .05, .30)
    texture_veto = density >= .16
    raw = 100*(.32*light + .32*flat + .14*low_texture + .22*axes)*(1-penalty)
    return min(raw, 49.) if texture_veto else raw, texture_veto


def score_rows(output, cutoff):
    target = output / 'scores.csv'
    if target.exists():
        raise FileExistsError(target)
    provenance = next(rows(output / 'provenance.csv'))
    if metadata_sha(output / 'features.csv') != provenance['features_sha256']:
        raise ValueError('Feature CSV differs from measurement provenance')
    doi = {row['path']: row['doi'].strip().casefold() for row in rows(DATASET / 'manifest.csv')}
    all_scores = []
    for row in rows(output / 'features.csv'):
        row['doi'] = doi[row['path']]
        for level in WHITE_LEVELS:
            row[f'white{level}_fraction'] = (
                int(row[f'white{level}_pixels']) / int(row['native_pixels']) if not row['error'] else ''
            )
        if row['error']:
            row.update(diagram_score='', texture_veto='', proposal='unclassified')
        else:
            score, veto = diagram_score(row)
            row.update(diagram_score=score, texture_veto=int(veto),
                       proposal='review' if score >= cutoff else 'below_review_cutoff')
        all_scores.append(row)
    if len(all_scores) != int(provenance['image_count']):
        raise ValueError('Unexpected feature count')
    return all_scores


def select_examples(scores):
    valid = [row for row in scores if not row['error']]
    rng = random.Random(42)
    rng.shuffle(valid)
    ranked = sorted(valid, key=lambda row: row['diagram_score'], reverse=True)
    used_sources = set()

    def sample(group, candidates):
        chosen = []
        for row in candidates:
            source = row['doi'] or row['path']
            if source in used_sources:
                continue
            used_sources.add(source)
            chosen.append(dict(review_id=f'{group[:3].upper()}-{len(chosen)+1:03d}', review_group=group, **row))
            if len(chosen) == EXAMPLES:
                return chosen
        raise ValueError(f'Only {len(chosen)} distinct sources available for {group}')

    top = sample('diagrammverdacht', (r for r in ranked if r['diagram_score'] > 20))
    lower = top[-1]['diagram_score']
    borderline = sample('grenzfaelle', (r for r in ranked if 20 < r['diagram_score'] < lower))
    sem = sample('wahrscheinlich_sem', (
        r for r in valid if r['diagram_score'] <= 20 and float(r['dense_texture_fraction']) >= .16
    ))
    return dict(diagrammverdacht=top, grenzfaelle=borderline, wahrscheinlich_sem=sem)


def render_sheets(groups, target):
    target.mkdir()
    names = {'diagrammverdacht': 'Bilder mit höchstem Diagrammverdacht', 'grenzfaelle': 'Grenzfälle',
             'wahrscheinlich_sem': 'Wahrscheinlich echte SEM-Aufnahmen'}
    title_font, text_font = (ImageFont.truetype('DejaVuSans.ttf', size=size) for size in (30, 15))
    for group, records in groups.items():
        pages = []
        for start in range(0, len(records), 25):
            index = start//25 + 1
            page = Image.new('RGB', (1350, 1250), '#eef1f4')
            draw = ImageDraw.Draw(page)
            draw.text((15, 12), f'{names[group]}  |  {index}/4', fill='#172137', font=title_font)
            draw.text((15, 52), 'MatSciFig SEM  |  Verdachtsscore, Weißanteil ab RGB 235 und Achsenpaar',
                      fill='#27374f', font=text_font)
            for offset, row in enumerate(records[start:start+25]):
                x, y = 8 + 270*(offset % 5), 81 + 232*(offset // 5)
                draw.rectangle((x, y, x+255, y+222), fill='white', outline='#cbd3dd')
                image = load_image(row['path'])
                image.thumbnail((243, 175), Image.Resampling.LANCZOS)
                page.paste(image, (x+(256-image.width)//2, y+6+(175-image.height)//2))
                white = 100*int(row['white235_pixels'])/int(row['native_pixels'])
                draw.text((x+6, y+183), f"{row['review_id']}    Score {row['diagram_score']:.1f}",
                          fill='#172137', font=text_font)
                draw.text((x+6, y+202), f"Weiß {white:.1f} %    Achsenpaar {'ja' if int(row['axis_pair']) else 'nein'}",
                          fill='#27374f', font=text_font)
            page.save(target / f'{group}_{index:02d}.jpg', quality=92, subsampling=0)
            pages.append(page)
        pages[0].save(target / f'{group}.pdf', resolution=120, save_all=True, append_images=pages[1:])


def report(output, cutoff):
    scored = score_rows(output, cutoff)
    groups = select_examples(scored)
    valid = [row for row in scored if not row['error']]
    thresholds = []
    for threshold in sorted(set(THRESHOLDS) | {cutoff}):
        count = sum(row['diagram_score'] >= threshold for row in valid)
        axes = sum(row['diagram_score'] >= threshold and int(row['axis_pair']) for row in valid)
        thresholds.append(dict(score_at_least=threshold, candidates=count,
                               fraction_of_all_images=count/len(scored), candidates_with_axis_pair=axes,
                               remaining_if_excluded=len(scored)-count, unclassified=len(scored)-len(valid)))
    write_csv(output / 'scores.csv', scored)
    write_csv(output / 'threshold_counts.csv', thresholds)
    write_csv(output / 'review_samples.csv', [row for records in groups.values() for row in records])
    render_sheets(groups, output / 'contact_sheets')
    method = (
        ('native_white_pixels', 'min(R,G,B) >= 225 / 235 / 245, counted at original resolution'),
        ('texture_analysis', 'Aspect-preserving area resize to <=384 px on long side, followed by grayscale'),
        ('large_homogeneous_regions', '9x9 local std <=3, 3x3 binary opening, 8-connected regions >=2% image area'),
        ('strong_texture', 'Fraction of pixels with 9x9 local gray-level std >=8'),
        ('dense_texture', '8x8 tile grid, tile qualifies at >=55% strongly textured pixels and <=40% gray235 pixels'),
        ('axis_pair', 'Canny + HoughLinesP, near horizontal/vertical runs each >=29% of its axis, bottom-left junction'),
        ('frame_suppression', 'Junction outside outer margins; reject a pair with both matching opposite sides'),
        ('axis_evidence', 'Minimum normalized line span times ramp of fraction gray >=235 inside plot field from .25 to .80'),
        ('score', '100*(.32*white+.32*flat+.14*low_texture+.22*axes)*(1-.65*ramp(dense_texture,.05,.30))'),
        ('white', 'ramp(.50*white225+.30*white235+.20*white245,.20,.80) using native fractions'),
        ('flat', 'ramp(large_flat_fraction,.30,.90)'),
        ('low_texture', '1-ramp(strong_texture_fraction,.06,.28)'),
        ('ramp', 'Linear interpolation from 0 at lower bound to 1 at upper bound, clamped to [0,1]'),
        ('texture_cap', 'At dense_texture_fraction >=.16, cap suspicion score to 49'),
        ('review_cutoff', str(cutoff)),
        ('contacts', '100 highest scores; 100 next highest strictly below first group; 100 texture-rich with score <=20. Sampling is independent of review cutoff.'),
        ('contact_diversity', 'Seed 42; at most one example per DOI across all contact sheets where DOI is present'),
        ('score_source_sha256', metadata_sha(Path(__file__))),
        ('interpretation', 'Unvalidated heuristic triage; scores are not probabilities or verified SEM/non-SEM labels'),
    )
    write_csv(output / 'method.csv', [dict(parameter=key, value=value) for key, value in method])
    print(json.dumps(thresholds, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('extract', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pixel-audit', type=Path, default=PIXEL_AUDIT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--review-threshold', type=float, default=REVIEW_THRESHOLD)
    args = parser.parse_args()
    if args.workers < 1 or not math.isfinite(args.review_threshold) or not 15 <= args.review_threshold <= 100:
        parser.error('workers must be positive; review threshold must be in [15, 100]')
    if args.stage == 'extract':
        extract(args.output, args.pixel_audit, args.workers)
    else:
        report(args.output, args.review_threshold)


if __name__ == '__main__':
    main()

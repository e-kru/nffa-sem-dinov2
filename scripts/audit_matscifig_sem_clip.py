"""Draw a DOI-balanced MatSciFig review sample and score it with pinned OpenAI CLIP."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.build_sem_posttraining_manifest import atomic_json, metadata_sha, rows

from src.config import DATASETS_ROOT


MODEL = 'ViT-B/32'
CLIP_COMMIT = 'dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1'
WEIGHTS_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
DATASET = DATASETS_ROOT / "matscifig_sem"
TRAINING_EXPORT = Path('outputs/sem_corpus/sem_posttraining_v0/training_manifest.csv')
SEED = 42
SAMPLE_SIZE = 400
MARGIN = .05
MIN_SIMILARITY = .25
REVIEW_FIELDS = ('review_id', 'image_id', 'path', 'source_file', 'doi', 'manual_label', 'notes')

PROMPTS = {
    'sem_or_mixed': (
        'a real scanning electron microscope SEM micrograph showing the texture of a material',
        'a grayscale scanning electron microscopy image of grains, surfaces, or fracture morphology',
        'a real SEM micrograph with a scale bar, measurement labels, arrows, or colored annotations',
        'a real electron microscope micrograph with colored outlines or a false color overlay showing the original texture',
        'a scientific figure containing a real scanning electron microscope image alongside diagrams or graphs',
        'a real SEM electron microscopy panel with inset images and letters',
    ),
    'plot': (
        'a scientific data graph with axes and a plotted curve, without a microscopy image',
        'a scientific chart, scatter plot, bar chart, or diffraction line plot with a legend and axes',
        'a graph of an EDS spectrum with peaks, numerical tick marks and text on a white background',
    ),
    'schematic': (
        'a drawn scientific schematic illustration with arrows and labeled shapes, no real microscopy photograph',
        'a cartoon diagram of a materials science process or laboratory equipment, without a micrograph',
        'a computer generated scientific rendering or drawing, not a photograph or microscope image',
    ),
    'phase_map': (
        'an artificial false color phase map with discrete flat color regions and no visible real SEM texture',
        'a synthetic segmentation mask or crystallographic orientation map with solid color regions and a color legend',
        'a simulated materials microstructure phase map with flat color domains, without a real microscopy image',
    ),
}


def write_csv(path, records, fields=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sample_candidates(export, dataset):
    local = {}
    for row in rows(dataset / 'manifest.csv'):
        if row['path'] in local:
            raise ValueError(f'Duplicate MatSciFig metadata row {row["path"]}')
        local[row['path']] = row
    eligible = []
    by_doi = defaultdict(list)
    seen_paths = set()
    for row in rows(export):
        if row['source'] != 'matscifig_sem' or row['included'] != '1':
            continue
        path = Path(row['path'])
        if path in seen_paths or not path.is_relative_to(dataset / 'images') or '..' in path.parts:
            raise ValueError(f'Invalid or duplicate MatSciFig candidate {path}')
        seen_paths.add(path)
        metadata = local[row['path']]
        if metadata['subtype_valid'].lower() != 'true':
            raise ValueError(f'Unexpected MatSciFig candidate subtype {path}')
        doi = metadata['doi'].strip().casefold()
        eligible.append(dict(image_id=row['image_id'], path=str(path), source_file=metadata['source_file'], doi=doi))
        if doi:
            by_doi[doi].append(eligible[-1])
    rng = random.Random(SEED)
    dois = rng.sample(sorted(by_doi), SAMPLE_SIZE)
    picks = [rng.choice(sorted(by_doi[doi], key=lambda row: row['path'])) for doi in dois]
    row_list = [dict(review_id=f'MSF-{index:03d}', **row, manual_label='', notes='')
                for index, row in enumerate(picks, 1)]
    return row_list, dict(eligible_images=len(eligible), publications=len(by_doi),
                          eligible_without_doi=sum(not row['doi'] for row in eligible))


def load_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def contact_sheets(records, output):
    output.mkdir()
    pages = []
    failures = []
    font = ImageFont.load_default(size=19)
    for start in range(0, len(records), 25):
        page = Image.new('RGB', (1300, 1210), '#eeeeee')
        draw = ImageDraw.Draw(page)
        draw.text((16, 12), f'MatSciFig SEM  |  Sheet {start//25+1:02d}  |  Review by ID',
                  fill='black', font=font)
        for offset, row in enumerate(records[start:start+25]):
            x, y = 10 + offset % 5 * 260, 45 + offset // 5 * 231
            try:
                image = load_image(row['path'])
                image.thumbnail((250, 197), Image.Resampling.LANCZOS)
                page.paste(image, (x + (250-image.width)//2, y + (197-image.height)//2))
            except (OSError, ValueError) as error:
                failures.append(dict(review_id=row['review_id'], error=str(error)))
                draw.text((x+10, y+85), 'Image load failed', fill='black', font=font)
            draw.text((x+5, y+201), row['review_id'], fill='black', font=font)
        page.save(output / f'sheet_{start//25+1:02d}.jpg', quality=90)
        pages.append(page)
    pages[0].save(output / 'contact_sheets.pdf', resolution=120, save_all=True, append_images=pages[1:])
    return failures


def prepare(output, dataset, training_export):
    records, population = sample_candidates(training_export, dataset)
    output.mkdir(parents=True)
    write_csv(output / 'review_sample.csv', records)
    failures = contact_sheets(records, output / 'contact_sheets')
    atomic_json(output / 'sample.json', population | dict(
        sample_images=len(records), unique_sample_dois=len({row['doi'] for row in records}),
        seed=SEED, sampling='uniform DOI without replacement, then uniform image within DOI',
        decode_failures=failures, training_export=str(training_export),
        training_export_sha256=metadata_sha(training_export), sample_sha256=metadata_sha(output / 'review_sample.csv')))
    print(json.dumps(population | {'sample_images': len(records), 'decode_failures': len(failures)}))


def proposal_for(views):
    """Both views must name the same exclusion class and clear both floors."""
    best = [max(view, key=view.get) for view in views]
    margins = [max(value for key, value in view.items() if key != 'sem_or_mixed') - view['sem_or_mixed']
               for view in views]
    if (len(set(best)) == 1 and best[0] != 'sem_or_mixed'
        and min(margins) >= MARGIN and min(view[best[0]] for view in views) >= MIN_SIMILARITY
    ):
        return 'review_for_exclusion', best[0]
    return 'keep', 'uncertain_or_sem_signal'


def priority_review(output, predictions):
    candidates = []
    fields = ('clip_review_class', 'weakest_margin_over_sem', *predictions[0])
    for record in predictions:
        if not record['full_sem_or_mixed']:
            continue
        views = [{key: float(record[f'{view}_{key}']) for key in PROMPTS} for view in ('full', 'center')]
        full, center = (max(view, key=view.get) for view in views)
        if full == center and full != 'sem_or_mixed':
            margin = min(view[full] - view['sem_or_mixed'] for view in views)
            candidates.append(dict(clip_review_class=full, weakest_margin_over_sem=round(margin, 6), **record))
    candidates.sort(key=lambda item: (-item['weakest_margin_over_sem'], item['review_id']))
    write_csv(output / 'priority_review.csv', candidates, fields)
    return candidates


def score(output, assets, threads):
    import torch

    sample = output / 'review_sample.csv'
    summary = json.loads((output / 'sample.json').read_text())
    if metadata_sha(sample) != summary['sample_sha256'] or (output / 'model_scores.csv').exists():
        raise ValueError('Use the unchanged blind sample and a fresh scoring run')
    weights = assets / 'weights/ViT-B-32.pt'
    if metadata_sha(weights) != WEIGHTS_SHA256:
        raise ValueError('Unexpected CLIP checkpoint SHA-256')
    sys.path[:0] = [str(assets / 'upstream' / f'CLIP-{CLIP_COMMIT}'), str(assets / 'deps')]
    import clip

    torch.set_num_threads(threads)
    model, preprocess = clip.load(str(weights), device='cpu', jit=False)
    model.eval()
    names = tuple(PROMPTS)
    prompts = [prompt for group in PROMPTS.values() for prompt in group]
    spans = []
    cursor = 0
    for group in PROMPTS.values():
        spans.append((cursor, cursor + len(group)))
        cursor += len(group)

    def group_scores(similarity):
        return {name: max(similarity[start:stop]) for name, (start, stop) in zip(names, spans)}

    records = list(rows(sample))
    if len(records) != SAMPLE_SIZE or tuple(records[0]) != REVIEW_FIELDS:
        raise ValueError('Unexpected review sample schema or length')
    predicted, started = [], time.monotonic()
    with torch.inference_mode():
        texts = torch.nn.functional.normalize(model.encode_text(clip.tokenize(prompts)).float(), dim=1)
        for start in range(0, len(records), 16):
            inputs, batch = [], records[start:start+16]
            for row in batch:
                try:
                    image = load_image(row['path'])
                    full = ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC, color='white')
                    inputs.append((preprocess(full), preprocess(image)))
                except (OSError, ValueError):
                    inputs.append(None)
            valid = [pair for pair in inputs if pair is not None]
            logits = []
            if valid:
                pixels = torch.stack([view for pair in valid for view in pair])
                features = torch.nn.functional.normalize(model.encode_image(pixels).float(), dim=1)
                logits = (features @ texts.T).reshape(len(valid), 2, len(prompts)).tolist()
            offset = 0
            for row, pair in zip(batch, inputs):
                value = {key: row[key] for key in REVIEW_FIELDS[:-2]}
                if pair is None:
                    value.update(proposal='keep', reason='decode_error', decode_ok=False)
                    for view in ('full', 'center'):
                        value.update({f'{view}_{name}': '' for name in names})
                else:
                    views = [group_scores(item) for item in logits[offset]]
                    offset += 1
                    decision, reason = proposal_for(views)
                    value.update(proposal=decision, reason=reason, decode_ok=True)
                    for view_name, item in zip(('full', 'center'), views):
                        value.update({f'{view_name}_{key}': f'{score:.6f}' for key, score in item.items()})
                predicted.append(value)
            if len(predicted) % 80 == 0:
                print(f'{len(predicted)}/{len(records)} images scored', flush=True)
    write_csv(output / 'model_scores.csv', predicted)
    proposals = [record for record in predicted if record['proposal'] == 'review_for_exclusion']
    write_csv(output / 'proposed_exclusions.csv', proposals, list(predicted[0]))
    priorities = priority_review(output, predicted)
    decode_failures = sum(not record['decode_ok'] for record in predicted)
    counts = dict(model=MODEL, upstream_commit=CLIP_COMMIT, weights_sha256=WEIGHTS_SHA256, prompts=PROMPTS,
        minimum_cosine=MIN_SIMILARITY, margin_over_sem=MARGIN, sample_images=len(records),
        scored_images=len(records)-decode_failures, decode_failures=decode_failures,
        proposed_exclusions=len(proposals), proposals_by_class=dict(Counter(row['reason'] for row in proposals)),
        priority_review_candidates=len(priorities),
        priority_review_by_class=dict(Counter(row['clip_review_class'] for row in priorities)),
        sample_keep=len(records)-len(proposals), actual_dataset_eligible=summary['eligible_images'],
        eligible_if_only_these_proposals_are_approved=summary['eligible_images']-len(proposals),
        manual_metrics=None, seconds=round(time.monotonic()-started, 2))
    atomic_json(output / 'model_summary.json', counts)
    print(json.dumps({key: value for key, value in counts.items() if key != 'prompts'}, indent=2))


def evaluate(output, labels_path):
    scores = {row['review_id']: row for row in rows(output / 'model_scores.csv')}
    labels = list(rows(labels_path))
    if len(labels) != len(scores) or {row['review_id'] for row in labels} != scores.keys():
        raise ValueError('Review CSV must preserve the full review-ID set')
    counts = Counter()
    for row in labels:
        label = row['manual_label'].strip().lower()
        scored = scores[row['review_id']]
        if row['image_id'] != scored['image_id'] or label not in ('', 'sem', 'non_sem', 'uncertain'):
            raise ValueError(f'Invalid manual review row: {row["review_id"]}')
        counts[label, scored['proposal']] += 1
    real_keep = counts['sem', 'keep']
    real_drop = counts['sem', 'review_for_exclusion']
    false_positive = counts['non_sem', 'keep']
    correct_drop = counts['non_sem', 'review_for_exclusion']
    report = dict(human_reviewed=sum(count for (label, _), count in counts.items() if label),
        sem_kept=real_keep, sem_proposed_for_exclusion=real_drop, non_sem_kept=false_positive,
        non_sem_proposed_for_exclusion=correct_drop,
        uncertain=sum(count for (label, _), count in counts.items() if label == 'uncertain'),
        sem_retention_recall=real_keep/(real_keep+real_drop) if real_keep+real_drop else None,
        exclusion_proposal_precision=correct_drop/(correct_drop+real_drop) if correct_drop+real_drop else None,
        interpretation='Reviewed, certain examples only; not a corpus-wide accuracy estimate.')
    if not report['human_reviewed']:
        raise ValueError('No manual review labels are available')
    atomic_json(output / 'manual_evaluation.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('sample', 'score', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--training-export', type=Path, default=TRAINING_EXPORT)
    parser.add_argument('--assets', type=Path, default=Path('outputs/dataset_audits/matscifig_clip_assets'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args()
    if args.stage == 'sample':
        prepare(args.output, args.dataset.absolute(), args.training_export)
    elif args.stage == 'score':
        score(args.output, args.assets.absolute(), args.threads)
    else:
        if args.labels is None:
            parser.error('evaluate requires --labels')
        evaluate(args.output, args.labels)


if __name__ == '__main__':
    main()

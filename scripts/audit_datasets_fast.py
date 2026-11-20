"""Bounded structural audit: filenames/short text previews, no pixel reads or hash computation."""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from src.config import DATASETS_ROOT


DATASETS = {
    'lithos': 211604, 'cross_microscope_steel': 5327, 'cigrocksem': 59816,
    'nffa_sem_100': 21169, 'uhcsdb': 961, 'microal': 7838, 'jlu_cathodes': 4112,
    'bil_nmc811': 486, 'bil_si_graphite': 5, 'kit_sem_powder': 24751,
    'aachen_heerlen': None,
}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp', '.jp2'}
TEXT_EXTENSIONS = {'.md', '.txt', '.rst', '.csv', '.tsv', '.json', '.xml', '.yaml', '.yml'}
TOKENS = {
    'segmentation_mask': {'mask', 'masks', 'groundtruth', 'ground_truth', 'gt', 'labelmap', 'labelmaps'},
    'derived_image': {'thumbnail', 'thumbnails', 'thumb', 'preview', 'overlay', 'overlays',
                      'montage', 'figure', 'figures', 'plot', 'plots', 'augmented', 'resized',
                      'tiles', 'tile', 'patches', 'patch', 'crops', 'crop', 'binary', 'skeleton'},
    'microscopy_image': {'images', 'image', 'micrograph', 'micrographs', 'original', 'originals',
                        'imgs', 'ppl', 'xpl', 'bse', 'se', 'sem', 'tem', 'hrtem',
                        'optical', 'electron', 'metallo', 'etd', 't2', 'rdbs'},
}
VARIANTS = {'PPL', 'XPL', 'SE', 'BSE', 'EBSD', 'EDS', 'EDX', 'BF', 'DF', 'SEM', 'TEM', 'HRTEM', 'ETD', 'T2', 'RDBS'}


def words(value):
    return set(re.findall(r'[a-z0-9]+', str(value).lower()))


def classify(relative, dataset=None):
    """Provisional filename-only categories. File suffix alone never proves microscopy."""
    path = Path(relative)
    tokens = words(relative)
    if '__MACOSX' in path.parts or path.name.startswith('._') or path.name == '.DS_Store':
        return 'metadata', 'OS sidecar/resource fork; not evidence of microscopy pixels'
    if path.suffix.lower() in {'.xlsx', '.xls', '.pdf'}:
        return 'metadata', 'spreadsheet/document container; contents not automatically read'
    if path.name.lower() in {'license', 'licence', 'readme'} or path.suffix.lower() in {'.html', '.htm'}:
        return 'metadata', 'document name/extension; no pixel content inspected'
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        if 'qa' in path.parts:
            return 'derived_image', 'QA preview context; kept separate from canonical images/masks'
        if dataset == 'kit_sem_powder' and 'Phenom_Images' in path.parts:
            return 'unknown', 'Phenom/BO/RL variants; microscopy versus label/derived role unconfirmed'
        if re.search(r'ground[_ -]?truth', str(relative), re.I):
            return 'segmentation_mask', 'explicit ground-truth path/name; pixel role unverified'
        if dataset == 'cross_microscope_steel' and re.match(r'\d+_[A-Za-z]+_(?:Bainite|Martensite)_\d+', path.stem, re.I):
            return 'derived_image', 'patch-number filename and dataset patch-metadata context; grouping unverified'
        for category in ('segmentation_mask', 'derived_image', 'microscopy_image'):
            if category == 'microscopy_image' and tokens & {'label', 'labels', 'segmentation', 'annotation', 'annotations'}:
                return 'unknown', 'image in ambiguous label/segmentation context; mask role unconfirmed'
            if tokens & TOKENS[category]:
                return category, 'filename/path token: '+','.join(sorted(tokens & TOKENS[category]))
        if dataset == 'nffa_sem_100' and re.fullmatch(r'L\d+_[0-9a-f]+', path.stem):
            return 'microscopy_image', 'NFFA class-folder/image-ID convention; SEM from existing project context, not pixels'
        return 'unknown', 'image-like extension; microscopy/mask/derived role unconfirmed'
    if path.suffix.lower() in TEXT_EXTENSIONS:
        if tokens & {'labels', 'label', 'annotation', 'annotations', 'classes', 'targets'} or path.name == 'center_component.csv':
            return 'label_or_annotation', 'text filename suggests labels/annotations; contents not fully validated'
        return 'metadata', 'text/table extension; only bounded previews may be read'
    return 'unknown', 'container, array or other file; contents not inspected'


def identifiers(relative, dataset=None):
    path = Path(relative)
    variants = sorted(v for v in VARIANTS if v.lower() in words(relative))
    sample = re.search(r'(?:^|[/_ -])(?:sample|specimen|section)(?:[_ -]+([A-Za-z0-9]+)|(\d+[A-Za-z0-9]*))(?=$|[/_. -])', str(path), re.I)
    tile = re.search(r'(.+?)[_-](?:tile|patch|crop)[_-]?(\d+(?:[_-]\d+)*)', path.stem, re.I)
    coordinates = re.search(r'(?:^|[_-])x[_-]?(\d+)[_-]y[_-]?(\d+)', path.stem, re.I)
    row = dict(variants=';'.join(variants) or 'unknown',
                sample_id_candidate=(sample.group(1) or sample.group(2)) if sample else 'unknown',
                original_image_id_candidate=tile.group(1) if tile else 'unknown',
                tile_candidate=tile.group(2) if tile else coordinates.group(0).strip('_-') if coordinates else 'unknown')
    if dataset == 'jlu_cathodes':
        match = re.search(r'(BM\d+)_SEM_', str(path))
        if match:
            row['sample_id_candidate'] = match.group(1)
        match = re.search(r'(?:BSE|SE)-(\d+)$', path.stem)
        if match and row['sample_id_candidate'] != 'unknown':
            row['original_image_id_candidate'] = row['sample_id_candidate']+'/slice-'+match.group(1)
    elif dataset == 'cross_microscope_steel':
        match = re.fullmatch(r'(\d+)_(.+_(?:Bainite|Martensite)_\d+)', path.stem, re.I)
        if match:
            row.update(original_image_id_candidate=match.group(2), tile_candidate=match.group(1))
    elif dataset == 'kit_sem_powder':
        match = re.fullmatch(r'Tile_r(\d+)_c(\d+)', path.stem)
        region = re.fullmatch(r'(Region\d+)_y(\d+)_x(\d+)_fw(.+)', path.stem)
        if match:
            row['tile_candidate'] = 'r'+match.group(1)+'_c'+match.group(2)
        elif region:
            row['tile_candidate'] = 'y'+region.group(2)+'_x'+region.group(3)+'_fw'+region.group(4)
        # Acquisition folder groups related tiles, but does not identify an original exposure.
    elif dataset == 'cigrocksem':
        plug = re.match(r'(plug_[A-Za-z0-9]+)_image_', path.stem)
        if plug:
            row['sample_id_candidate'] = plug.group(1)
        original = re.split(r'_(?:crop|resizeTo|top|bottom|up|down)(?:_|\b)', path.stem, maxsplit=1)[0]
        if original != path.stem:
            row['original_image_id_candidate'] = original
            row['tile_candidate'] = path.stem[len(original):].strip('_')
    elif dataset == 'microal':
        match = re.fullmatch(r'(.+)_(\d+)_(\d+)', path.stem)
        if match and 'Metallo' in path.parts:
            row.update(original_image_id_candidate=match.group(1), tile_candidate=match.group(2)+'_'+match.group(3))
    return row


def metadata_candidate(name):
    lower = name.lower()
    return (Path(lower).suffix in TEXT_EXTENSIONS and
            (lower.startswith(('readme', 'license', 'description', 'metadata', 'info', 'dataset', 'citation'))
             or Path(lower).suffix in {'.csv', '.tsv', '.json'}))


def scan_dataset(root, dataset, allowance, *, entries_per_directory=24, max_directories=8,
                 max_depth=7, max_examples=200, preview_bytes=4096, max_previews=3, start_paths=None):
    base = Path(root)/dataset
    queue = deque([(base/relative, len(Path(relative).parts)) for relative in start_paths] if start_paths else [(base, 0)])
    files, directories, previews, errors = [], [], [], []
    inspected = 0
    while queue and inspected < allowance and len(directories) < max_directories and len(files) < max_examples:
        directory, depth = queue.popleft()
        selected, children = [], []
        limit = min(entries_per_directory, allowance-inspected)
        try:
            with os.scandir(directory) as entries:
                for _ in range(limit):
                    try:
                        entry = next(entries)
                    except StopIteration:
                        break
                    inspected += 1
                    relative = str(Path(entry.path).relative_to(base))
                    if entry.is_symlink():
                        selected.append(dict(path=relative, kind='symlink_not_followed'))
                    elif entry.is_dir(follow_symlinks=False):
                        selected.append(dict(path=relative, kind='directory'))
                        children.append((Path(entry.path), depth+1))
                    elif entry.is_file(follow_symlinks=False) and len(files) < max_examples:
                        category, evidence = classify(relative, dataset)
                        row = dict(dataset=dataset, relative_path=relative, category=category,
                                   category_evidence=evidence, **identifiers(relative, dataset))
                        files.append(row)
                        selected.append(dict(path=relative, kind='file'))
                        if len(previews) < max_previews and metadata_candidate(entry.name):
                            try:
                                # No full-file read or image decoding, even for unexpectedly large tables.
                                with Path(entry.path).open('rb') as stream:
                                    content = stream.read(preview_bytes)
                                previews.append(dict(path=relative, bytes_read=len(content),
                                                     text=content.decode('utf-8', errors='replace'),
                                                     possibly_truncated=len(content) == preview_bytes))
                            except OSError as error:
                                errors.append(f'{relative}: {error}')
                    if len(files) >= max_examples:
                        break
            directories.append(dict(path=str(directory.relative_to(base)), depth=depth,
                                    prefix_limit=limit, prefix_only=len(selected) >= limit, entries=selected))
            if depth < max_depth:
                # Prioritize explicit roles within the already bounded prefix, never sort a full directory.
                children.sort(key=lambda item: (not bool(words(item[0].name) &
                    {'images', 'imgs', 'masks', 'labels', 'annotations', 'raw', 'originals', 'metadata'}), item[0].name))
                children = [child for child in children if child[0].name not in {'__MACOSX', 'download', '.git', '__pycache__'}]
                queue.extendleft(reversed(children[:3]))
        except OSError as error:
            errors.append(f'{directory}: {error}')
            directories.append(dict(path=str(directory.relative_to(base)), depth=depth, entries=[], error=str(error)))
    return dict(dataset=dataset, inspected_entries=inspected, files=files, directories=directories,
                metadata_previews=previews, errors=errors, complete=False,
                entry_budget_reached=inspected >= allowance,
                limitations='Bounded directory prefixes in filesystem order; no pixel inspection or full inventory. '
                            'Unvisited siblings/deeper paths may contain different roles or variants.')


def summary(result):
    files = result['files']
    categories = Counter(row['category'] for row in files)
    image_paths = sorted({str(Path(r['relative_path']).parent) for r in files
        if r['category'] == 'microscopy_image' or (r['category'] == 'derived_image' and
            ('imgs' in Path(r['relative_path']).parts or result['dataset'] == 'cross_microscope_steel'))})
    mask_paths = sorted({str(Path(r['relative_path']).parent) for r in files if r['category'] == 'segmentation_mask'})
    variants = sorted({v for r in files for v in r['variants'].split(';') if v != 'unknown'})
    context = ' '.join(p['text'] for p in result['metadata_previews'])
    for variant in VARIANTS:
        if re.search(r'\b'+variant+r'\b', context, re.I):
            variants.append(variant)
    variants = sorted(set(variants))
    optical = bool(set(variants) & {'PPL', 'XPL'})
    electron = bool(set(variants) & {'SE', 'BSE', 'EBSD', 'EDS', 'EDX', 'SEM', 'TEM', 'HRTEM'})
    row = dict(dataset=result['dataset'], reported_image_count=DATASETS[result['dataset']],
                reported_count_source='user_prior_file_count; not recounted; not a verified usable-image count',
                modality='polarized optical microscopy (name/metadata hint)' if optical else 'electron microscopy (name/metadata hint)' if electron else 'unknown',
                material='unknown', identified_image_paths=';'.join(image_paths) or 'unknown',
                identified_mask_paths=';'.join(mask_paths) or 'unknown',
                sample_id_available='candidate' if any(r['sample_id_candidate'] != 'unknown' for r in files) else 'unknown',
                original_image_id_available='candidate' if any(r['original_image_id_candidate'] != 'unknown' for r in files) else 'unknown',
                tile_structure='filename candidates; verify grouping' if any(r['tile_candidate'] != 'unknown' for r in files) else 'unknown',
                variants=';'.join(variants) or 'unknown', inspected_entries=result['inspected_entries'],
                inspected_directories=len(result['directories']), example_files=len(files),
                sampled_categories=json.dumps(categories, sort_keys=True), scan_complete=False,
                uncertainties='No semantic/pixel validation; sample/acquisition identity and file-role rules require confirmation; '
                              + result['limitations'] + (' Errors: '+'; '.join(result['errors']) if result['errors'] else ''))
    interpretation = interpret(result)
    row.update(interpretation['fields'])
    row['uncertainties'] = ' '.join(interpretation['notes'])+' '+row['uncertainties']
    row['identified_annotation_paths'] = ';'.join(sorted({str(Path(r['relative_path']).parent)
        for r in files if r['category'] == 'label_or_annotation'})) or 'unknown'
    row['identified_derived_image_paths'] = ';'.join(sorted({str(Path(r['relative_path']).parent)
        for r in files if r['category'] == 'derived_image'})) or 'unknown'
    observed = {e['path'] for d in result['directories'] for e in d['entries'] if e['kind'] == 'directory'}
    row['observed_mask_directory_candidates'] = ';'.join(sorted(p for p in observed
        if Path(p).name.lower() in {'mask', 'masks'})) or 'unknown'
    return row


def interpret(result):
    """Qualified findings gated on sampled evidence; no additional filesystem access."""
    dataset = result['dataset']
    paths = '\n'.join(r['relative_path'] for r in result['files'])
    observed = '\n'.join(e['path'] for d in result['directories'] for e in d['entries'])
    metadata = '\n'.join(p['text'] for p in result['metadata_previews'])
    fields, notes = {}, []
    if dataset == 'lithos' and 'ppl-0.png' in paths:
        fields.update(material='Gestein/Minerale (PPL/XPL und lokale Mineral-Labeltabelle)',
                      tile_structure='GMAS_LOTE*/<number>/<number> with paired ppl/xpl; hierarchy semantics unknown')
        notes += ['PPL/XPL liegen paarweise im selben numerischen Blattordner; Registrierung wurde nicht geprüft. '
                  'Die Zahlenordner sind Gruppierungskandidaten, aber Probe, Schliff, Original und Patch sind noch nicht eindeutig zugeordnet. '
                  'Fold-CSV enthält Sec_patch-Pfade; daraus werden keine neuen Splits erzeugt.']
        if 'Multiclass_label_name' in metadata:
            notes += ['center_component.csv enthält Mineralnamen, Klassen, Achsen und Annotation-/Schnittkoordinaten; '
                      'das sind Labels/Annotationen, kein Nachweis einer Rastermaske. GMAS_LOTE2 wurde nur als Ordner gesehen.']
    elif dataset == 'cross_microscope_steel' and 'Bainite' in paths:
        fields.update(material='Stahl; Bainit/Martensit (Pfadnamen)',
                      tile_structure='<patch number>_<A_phase_original number>.png (candidate)')
        notes += ['Beispiel 38_A_Bainite_01.png: 38 ist Patch-Kandidat, A_Bainite_01 Originalgruppen-Kandidat. '
                  'A ist keine hier bestätigte Mikroskop-/Proben-ID. Dataset_Patches_Microscope_Metadata.xlsx wurde gefunden, '
                  'aber nicht eingelesen; Modalitäten, Proben und Patch-Eltern müssen dort später abgeglichen werden.']
    elif dataset == 'cigrocksem' and '/imgs/' in paths and '/masks/' in paths:
        fields.update(material='Gestein: mudstone/sandstone/shale (Ordner)',
                      modality='SEM (Datensatzname/Kontext; nicht pixelvalidiert)',
                      tile_structure='crop/resizeTo/quadrant suffixes; plug/image prefixes (candidates)')
        notes += ['raw/data/{mudstone,sandstone,shale}/{imgs,masks} trennt Bild- und Maskenrollen strukturell. '
                  'Gleiche Namen in den untersuchten Paarordnern bestätigen nur die Namenskonvention, nicht Geometrie oder Klassen. '
                  'plug_44/plug_47A, image-Nummern sowie andere Stammpräfixe sind Proben-/Originalkandidaten; '
                  'Crop-, Quadranten- und Resize-Varianten können denselben Ursprung teilen.']
    elif dataset == 'nffa_sem_100' and re.search(r'L\d+_[0-9a-f]+\.jpg', paths):
        fields.update(material='gemischt; Morphologie-/Anwendungsklassen (Ordner)',
                      modality='SEM (bestehender Projektkontext)')
        notes += ['Klassen liegen direkt unter dem Datensatz, z. B. Biological/Fibres/Particles. '
                  'L7_<hex>.jpg ist ein Bild-/Klassenbezeichner, keine bestätigte Proben- oder Akquisitions-ID. '
                  '21.169 ist die übernommene Rohdatei-Vorzählung, nicht der Umfang des kuratierten NFFA-V3-Manifests; '
                  'bestehende Metadatenstreifen-Behandlung, Manifeste und Leakage-Gruppen bleiben unverändert.']
    elif dataset == 'uhcsdb' and 'sample_key' in metadata:
        fields.update(material='ultrahochkohlenstoffhaltiger Stahl (lokales README/.zenodo)',
                      modality='SEM; SE/BSE/unknown (lokales README, nicht nachgezählt)',
                      sample_id_available='metadata.csv sample_key (lokales README; Join noch ungeprüft)')
        notes += ['README nennt 961 Mikrographien aus 43 Proben sowie SE/BSE/unknown; dies sind lokale Quellenangaben, '
                  'keine hier nachgezählten Verteilungen. metadata.csv.path soll auf micrographs/ verweisen; '
                  'sample_key ist der spätere Proben-Gruppierungsschlüssel, nicht micrograph1001. '
                  'Der Tabelleninhalt und vollständige Join wurden hier nicht geprüft. '
                  'Vorhandene Hashwerte im kurzen manifest.json-Textauszug wurden lediglich gelesen, nie neu berechnet.']
    elif dataset == 'microal' and 'Electron/' in paths:
        fields.update(material='metallografisch/gemischt; konkrete Legierungen unknown',
                      modality='SEM/TEM/HRTEM + Metallo; optische Modalität dort unknown (Ordner)',
                      tile_structure='Metallo filename prefix + trailing numeric pairs (candidates)')
        notes += ['Electron trennt SEM, TEM, TEM dark-field und HRTEM; Metallo enthält u. a. Heat-treated. '
                  'image/image_test und label/label_test sind getrennte Rollenordner; passende Dateinamen wurden in image/label gesehen. '
                  'Ob label-Pixel Segmentklassen, Kanten oder andere Ziele sind, bleibt unknown. '
                  'Ein mask-Ordner wurde gesehen, dessen Inhalt jedoch nicht untersucht. '
                  'TEM annotation/train.json und test.json wurden nur kurz als Metadaten gelesen; keine Testbilder geöffnet. '
                  'Deprecated repository enthält auch .pth-Dateien; diese sind keine Bildbeispiele. '
                  'MicroAl bedeutet hier nicht automatisch Aluminium.']
    elif dataset == 'jlu_cathodes' and 'slice-In-Beam' in paths:
        fields.update(material='Kathoden (Datensatzkontext); Chemie unknown', modality='SEM; SE/BSE (Ordner/Dateinamen)',
                      tile_structure='serial slice numbers; not established 2D tile grid')
        notes += ['BM01/BM03/BM10 sind Proben-/Volumengruppen-Kandidaten. SE und BSE teilen Gruppen- und Slice-Nummern; '
                  'Paarung/Registrierung ist ungeprüft. slice-...-00001 weist auf eine Schnittserie hin; '
                  'Nachbarslices sind nicht automatisch unabhängige Originalaufnahmen.']
    elif dataset == 'bil_nmc811' and 'img_RDBS_' in paths:
        fields.update(material='NMC811 (Datensatzname; nicht chemisch verifiziert)', modality='SEM (sem_images-Pfad)',
                      tile_structure='sequential filenames; stack versus independent images unknown')
        notes += ['RDBS ist als wörtliche Dateinamenvariante erfasst, nicht physikalisch interpretiert. '
                  'Fortlaufende img_RDBS_0000-Nummern geben ohne Metadaten keine Proben-, Volumen- oder Akquisitionsgrenzen an.']
    elif dataset == 'bil_si_graphite' and 'Electron Image BSE' in paths:
        fields.update(material='Si/Graphit (Datensatzname; nicht chemisch verifiziert)', modality='Elektronenmikroskopie; BSE/ETD/T2 (Dateinamen)')
        notes += ['Gleicher Electron Image BSE-Stamm liegt als PNG und TIFF vor: Formatduplikat-Kandidat, keine bestätigte Pixelgleichheit. '
                  'ANO-BSE/ETD/T2 sowie Spannung/Strom sind Variantenhinweise; gleiche Probe/Position ist ungeprüft. '
                  'Fünf Dateien sind deshalb nicht nachweislich fünf unabhängige Aufnahmen.']
    elif dataset == 'kit_sem_powder' and 'sem_segmentation_ssl' in observed:
        fields.update(material='Pulver (Datensatzkontext); Zusammensetzung unknown',
                      modality='SEM-Kontext; Rollen der BO/RL/Phenom-Varianten unknown',
                      tile_structure='acquisition folder + Tile_rNNN_cNNN / RegionNNN_yNN_xNN_fw... (candidates)')
        notes += ['Tile_r000_c000 wiederholt sich in verschiedenen Akquisitionsordnern; der Dateistamm allein ist kein globaler Originalschlüssel. '
                  'Region-/y/x-/Feldweiten-Namen liefern weitere Gruppierungshinweise. '
                  'BO/RL/Phenom können Bild-/Masken-/Ableitungsrollen haben und bleiben unknown.']
        names = {split: {Path(r['relative_path']).name for r in result['files']
                         if f'/{split}/Phenom_Images/' in r['relative_path']} for split in ('train', 'val')}
        overlap = sorted(names['train'] & names['val'])
        if overlap:
            notes += ['In train/Phenom_Images und val/Phenom_Images stimmen '+str(len(overlap))+
                      ' untersuchte Dateinamen überein (z. B. '+', '.join(overlap[:3])+'). '
                      'Das ist ein Überlappungshinweis, kein Hash-/Pixelduplikatnachweis; vor späterer Nutzung gezielt prüfen. '
                      'Die vorhandenen Splits werden nicht verändert.']
    elif dataset == 'aachen_heerlen' and 'images/' in paths and 'masks/' in paths:
        fields.update(material='Stahl (lokale Metadatennamen/Projektkontext)', modality='SEM (bestehender Projektkontext)')
        notes += ['images/ und masks/ sind getrennt, mit passenden IMG_*-Namen in der Stichprobe. '
                  'qa/ enthält abgeleitete Maskenvorschauen/Overlays und gehört nicht zu den kanonischen Bild-/Maskenpaaren. '
                  'Polygon-CSV hat image_url/point/polygon; Metadaten haben type/temperature/direction/distance/magnification/angle. '
                  'Diese Felder liefern Gruppierungshinweise, aber IMG_00001 allein ist keine Proben-ID. '
                  'Bestehende Gruppierung und Holdouts bleiben unverändert; PNG.zip bleibt ungeöffnet.']
    return dict(fields=fields, notes=notes)


def write_csv(path, rows, fields=None):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(results, summaries, budget):
    lines = ['# Schneller struktureller Dataset-Audit', '',
             'Zeitpunkt: '+datetime.now(timezone.utc).isoformat(), '',
             f'Globales Budget: {budget} Verzeichniseinträge; tatsächlich gelesen: '+str(sum(r['inspected_entries'] for r in results))+'.',
             'Alle Zählungen betreffen ausschließlich untersuchte Beispiele. Vorzählungen stammen vom Nutzer, '
             'nicht aus einer neuen Vollzählung und nicht aus einer hier verifizierten Publikation.',
             'Keine Bildpixel geöffnet, keine Hashberechnungen, keine GPU, keine Trainingsauswahl/Splits und keine Rohdatenänderungen.', '',
             '## Methode und Reproduktion', '',
             'Nur begrenzte os.scandir-Prefixe (24 Einträge je Verzeichnis, höchstens 8 Verzeichnisse je Datensatz '
             'und Scan-Pass, Tiefe höchstens 7); Dateisystemreihenfolge statt repräsentativer Zufallsstichprobe. '
             'Standardmäßig höchstens 180 Einträge (konfigurierbar bis 200) und 200 Beispieldateien je Datensatz, global höchstens 2.000 Einträge. '
             'Ein optionaler Fortsetzungspass besucht nur zuvor beobachtete, noch nicht gelesene Unterordner '
             'und zieht frühere Einträge vom selben Budget ab. Symlinks werden nicht verfolgt. '
             'Es werden höchstens 3 Metadaten-Vorschauen je Datensatz/Scan-Pass mit je 4 KiB gelesen; '
             'das kann nur den Tabellenkopf enthalten. PDFs, XLSX, Bilddateien und Archive werden nicht geöffnet. '
             'Die Befunde sind durchgehend unvollständig; nicht beobachtet bedeutet nicht abwesend.', '',
             'Neuer begrenzter Audit (bestehendes Ziel wird nie überschrieben):',
             '```bash', '# Run from the repository root',
             '/usr/bin/python3 scripts/audit_datasets_fast.py --output /tmp/dataset_audit_new', '```',
             'Nur Berichte aus vorhandener Evidenz neu erzeugen, ohne Datensatz-Zugriffe:',
             '```bash', '/usr/bin/python3 scripts/audit_datasets_fast.py \\',
             '  --render-from data/_audit/audit_evidence.json \\',
             '  --output /tmp/dataset_audit_rendered', '```', '',
             '## Vorläufige Übersicht', '',
             '| Dataset | Vorzählung | Einträge / Beispieldateien | Modalität | Bild-/Maskenpfade* | Varianten |',
             '| --- | ---: | ---: | --- | --- | --- |']
    for row in summaries:
        lines.append(f"| {row['dataset']} | {row['reported_image_count'] if row['reported_image_count'] is not None else 'unknown'} | "
                     f"{row['inspected_entries']} / {row['example_files']} | {row['modality']} | "
                     f"{len(row['identified_image_paths'].split(';')) if row['identified_image_paths'] != 'unknown' else 'unknown'} / "
                     f"{len(row['identified_mask_paths'].split(';')) if row['identified_mask_paths'] != 'unknown' else 'unknown'} | {row['variants']} |")
    lines += ['', '## Grenzen der Kategorien', '',
              '*Pfadanzahlen betreffen nur belegte Rollen in der Stichprobe; konkrete Pfade stehen in der CSV und unten. '
              'Bildpfade können bereits geschnittene Mikroskopie-Patches enthalten; volle Originalaufnahmen sind damit nicht nachgewiesen.', '',
              '`microscopy_image` und `segmentation_mask` sind vorläufige Pfad-/Namenshinweise, keine Pixelvalidierung. '
              'Ein Bildsuffix allein bleibt `unknown`. Explizite Tiles/Crops/Overlays gelten als `derived_image`; '
              'sie können Mikroskopie-Inhalt enthalten. Labels in Texttabellen sind `label_or_annotation`. '
              'Array-/Archivformate bleiben ungeöffnet und `unknown`. Aus fehlenden Maskenbeispielen folgt nicht, '
              'dass ein Datensatz keine Masken enthält. Ein Dateistamm ist nicht automatisch eine Proben-/Aufnahme-ID.', '']
    for result, row in zip(results, summaries):
        lines += ['## '+result['dataset'], '',
                  f"Unvollständiger Befund: {result['inspected_entries']} Einträge, {len(result['directories'])} Verzeichnisse, {len(result['files'])} Beispieldateien.",
                  'Material: '+row['material']+'; Modalität: '+row['modality']+'.',
                  'Proben-ID: '+row['sample_id_available']+'; Originalaufnahme-ID: '+row['original_image_id_available']+'; Tiles: '+row['tile_structure']+'.', '',
                  'Bildpfade: `'+row['identified_image_paths']+'`.',
                  'Maskenpfade mit Dateibeispielen: `'+row['identified_mask_paths']+'`.',
                  'Beobachtete Maskenordner (Inhalt ggf. ununtersucht): `'+row['observed_mask_directory_candidates']+'`.', '',
                  *('- '+note for note in interpret(result)['notes']), '',
                  'Untersuchte Verzeichnisse (jeweils begrenzter Prefix):', '']
        for directory in result['directories']:
            entries = directory['entries']
            names = ', '.join(e['path'].replace('|', '\\|')+('/' if e['kind'] == 'directory' else '') for e in entries[:8])
            lines.append(f"- `{directory['path']}`: {names or directory.get('error', 'leer')}")
        lines += ['', 'Beispielnamen und vorläufige Rollen:', '']
        for sample in result['files'][:12]:
            lines.append(f"- `{sample['relative_path']}` → `{sample['category']}`; {sample['category_evidence']}.")
        if result['metadata_previews']:
            lines += ['', 'Gezielt gelesene Metadaten-Vorschauen (höchstens 4 KiB je Datei):', '']
            for preview in result['metadata_previews']:
                lines += [f"**`{preview['path']}`**, {preview['bytes_read']} Bytes; vollständige Datei nicht zugesichert:",
                          '```text', preview['text'].replace('```', "'''"), '```', '']
        lines += ['Grenzen: '+result['limitations']+(' Fehler: '+'; '.join(result['errors']) if result['errors'] else ''), '']
    lines += ['## Gezielte nächste Prüfungen für späteres Cleaning', '',
              '1. README/Metadaten gegen Dateinamenregeln prüfen; Bild-/Maskenrollen an wenigen passenden Paaren visuell und geometrisch bestätigen.',
              '2. Proben-, Akquisitions-, Slice-/Tile- und Modalitätszuordnung dokumentieren; Dateinummern nicht mit unabhängigen Proben verwechseln.',
              '3. PPL/XPL, SE/BSE und abgeleitete Varianten paaren; Registrierung und gemeinsame Ursprungsaufnahme überprüfen.',
              '4. Verborgene Rollen in Archiven, HDF5/MAT/NPY und großen Metadatentabellen gezielt untersuchen, bevor ein Vollmanifest entsteht.',
              '5. Erst in einem eigenen Cleaning-Auftrag Formate, Bildqualität, Metadatenstreifen, Duplikate und Benchmark-Überlappung prüfen. '
              'Bestehende NFFA-/Aachen-Holdouts und Masken erhalten.', '',
              'Besonders gezielt: KIT train/val-Namensüberschneidungen und BO/RL-Rollen; MicroAl label/mask-Semantik; '
              'LITHOS Zahlenhierarchie; Cross-Microscope XLSX-Zuordnung; UHCS sample_key-Join; '
              'CIGRockSEM Crop-/Resize-Ursprünge; JLU Slice-/Kanal-Paarung; BIL Format-/Serienvarianten.', '',
              'Dieser Audit führt keinen dieser vollständigen Cleaning-Schritte aus.', '']
    return '\n'.join(lines)


def write_artifacts(output, results, budget):
    """Render existing evidence without opening any source image or dataset directory."""
    for result in results:
        if len(result['files']) > 200:
            raise ValueError('Evidence exceeds the filename budget')
        for row in result['files']:
            row['category'], row['category_evidence'] = classify(row['relative_path'], result['dataset'])
            row.update(identifiers(row['relative_path'], result['dataset']))
    inspected = sum(r['inspected_entries'] for r in results)
    if inspected > budget:
        raise ValueError('Evidence exceeds the global entry budget')
    summaries = [summary(result) for result in results]
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'dataset_summary.csv', summaries)
    files = [row for result in results for row in result['files']]
    write_csv(output/'sample_manifest.csv', files, fields=['dataset', 'relative_path', 'category', 'category_evidence',
              'variants', 'sample_id_candidate', 'original_image_id_candidate', 'tile_candidate'])
    (output/'structure_report.md').write_text(report_text(results, summaries, budget))
    (output/'audit_evidence.json').write_text(json.dumps(dict(results=results, summaries=summaries,
        inspected_entries=inspected, budget=budget), indent=2)+'\n')
    print('Audit artifacts: '+str(output), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DATASETS_ROOT)
    p.add_argument('--output', type=Path, default=DATASETS_ROOT / "_audit")
    p.add_argument('--max-entries', type=int, default=2000)
    p.add_argument('--per-dataset-entries', type=int, default=180)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--continue-from', type=Path, help='Continue only unvisited observed directories; reuse the existing global budget')
    mode.add_argument('--render-from', type=Path, help='Only regenerate reports from existing evidence; do not access datasets')
    args = p.parse_args(argv)
    if not 1 <= args.max_entries <= 2000 or not 1 <= args.per_dataset_entries <= 200:
        p.error('Use at most 2000 total entries and 200 per dataset')
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing audits are never overwritten')
    if args.render_from:
        evidence = json.loads(args.render_from.read_text())
        write_artifacts(args.output, evidence['results'], min(args.max_entries, evidence['budget']))
        return
    previous = json.loads(args.continue_from.read_text())['results'] if args.continue_from else []
    prior = {result['dataset']: result for result in previous}
    results, remaining = [], args.max_entries-sum(r['inspected_entries'] for r in previous)
    if remaining < 0:
        p.error('Prior audit already exceeds the requested budget')
    for dataset in DATASETS:
        old = prior.get(dataset)
        if old:
            visited = {d['path'] for d in old['directories']}
            frontier = [entry['path'] for d in old['directories'] for entry in d['entries']
                        if entry['kind'] == 'directory' and entry['path'] not in visited
                        and not set(Path(entry['path']).parts) & {'__MACOSX', 'download', '.git', '__pycache__'}]
            frontier.sort(key=lambda path: (not bool(words(path) & {'images', 'imgs', 'masks'}), path))
            allowance = max(0, min(args.per_dataset_entries-old['inspected_entries'], remaining))
            result = scan_dataset(args.root, dataset, allowance, start_paths=frontier,
                                  max_examples=max(0, 200-len(old['files']))) if frontier else None
            if result:
                remaining -= result['inspected_entries']
                for key in ('files', 'directories', 'metadata_previews', 'errors'):
                    old[key].extend(result[key])
                old['inspected_entries'] += result['inspected_entries']
                old['entry_budget_reached'] = old['inspected_entries'] >= args.per_dataset_entries
            result = old
        else:
            result = scan_dataset(args.root, dataset, min(args.per_dataset_entries, remaining))
            remaining -= result['inspected_entries']
        results.append(result)
        print(json.dumps(dict(dataset=dataset, entries=result['inspected_entries'], examples=len(result['files']))), flush=True)
    write_artifacts(args.output, results, args.max_entries)


if __name__ == '__main__':
    main()

import csv
import json

import pytest

from scripts import audit_matscifig_sem_clip as audit


def test_doi_sampling_is_reproducible_and_never_selects_held_out_rows(tmp_path, monkeypatch):
    dataset = tmp_path / 'matscifig_sem'
    dataset.mkdir()
    monkeypatch.setattr(audit, 'SAMPLE_SIZE', 4)
    local, export = [], []
    for doi in range(5):
        for panel in range(2):
            path = str(dataset / 'images' / f'{doi}_{panel}.jpg')
            local.append(dict(path=path, source_file=f'{doi}_{panel}.jpg', doi=f'doi:{doi}', subtype_valid='True'))
            export.append(dict(path=path, image_id=f'{doi}_{panel}', source='matscifig_sem',
                               included='0' if doi == 4 else '1'))
    audit.write_csv(dataset / 'manifest.csv', local)
    audit.write_csv(tmp_path / 'train.csv', export)
    sample, counts = audit.sample_candidates(tmp_path / 'train.csv', dataset)
    assert counts['eligible_images'] == 8
    assert {row['doi'] for row in sample} == {f'doi:{doi}' for doi in range(4)}
    assert sample == audit.sample_candidates(tmp_path / 'train.csv', dataset)[0]
    assert all(not row['manual_label'] for row in sample)


def test_proposal_requires_consensus_strong_signal_and_clear_sem_margin():
    chart = dict(sem_or_mixed=.20, plot=.30, schematic=.16, phase_map=.10)
    mixed = chart | {'sem_or_mixed': .28}
    schematic = chart | {'schematic': .33}
    weak = chart | {'plot': .24, 'sem_or_mixed': .16}
    assert audit.proposal_for([chart, chart]) == ('review_for_exclusion', 'plot')
    for view in (mixed, schematic, weak):
        assert audit.proposal_for([chart, view])[0] == 'keep'


def test_blank_review_must_not_create_accuracy_report(tmp_path):
    audit.write_csv(tmp_path / 'model_scores.csv', [
        dict(review_id='MSF-001', image_id='fixture', proposal='keep'),
    ])
    labels = tmp_path / 'labels.csv'
    audit.write_csv(labels, [dict(review_id='MSF-001', image_id='fixture', manual_label='')])
    with pytest.raises(ValueError, match='No manual review'):
        audit.evaluate(tmp_path, labels)
    assert not (tmp_path / 'manual_evaluation.json').exists()
    audit.write_csv(labels, [dict(review_id='MSF-001', image_id='fixture', manual_label='SEM')])
    audit.evaluate(tmp_path, labels)
    with (tmp_path / 'model_scores.csv').open(newline='') as handle:
        assert next(csv.DictReader(handle))['proposal'] == 'keep'
    assert json.loads((tmp_path / 'manual_evaluation.json').read_text())['sem_retention_recall'] == 1

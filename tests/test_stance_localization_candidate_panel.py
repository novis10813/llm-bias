from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from pathlib import Path

import pytest

from llm_bias.core.artifact_paths import sha256_json
from llm_bias.core.stance_localization_candidate_panel import (
    LocalizationCandidatePanel, build_localization_candidate_panel,
    HISTORICAL_SOURCE, HISTORICAL_SOURCE_SHA256, _candidate_layers,
)

CASES = (
    ('qwen3.5-4b', 32, (0, 3, 14, 15, 16, 31), 72384, 15, 16),
    ('glm4-9b-0414', 40, (0, 4, 19, 20, 21, 39), 72384, 20, 19),
    ('gemma4-12b-it', 48, (0, 5, 23, 26, 27, 28, 47), 84448, 27, 27),
    ('gpt-oss-20b', 24, (0, 2, 7, 8, 9, 11, 14, 23), 96512, 8, 14),
)


@pytest.mark.parametrize('slug,count,layers,cells,peak,injection', CASES)
def test_exact_panel_and_provenance(slug, count, layers, cells, peak, injection):
    panel = build_localization_candidate_panel(model_slug=slug, actual_layer_count=count)
    assert panel.layers == layers
    assert panel.development_cell_count == cells == 3016 * 4 * len(layers)
    record = panel.to_dict()
    assert record['historical_peak'] == peak
    assert record['historical_injection_layer'] == injection
    assert record['actual_layer_count'] == record['expected_layer_count'] == count
    assert record['historical_source'] == HISTORICAL_SOURCE
    assert record['historical_source_sha256'] == HISTORICAL_SOURCE_SHA256
    assert record['site'] == 'post' and record['selector'] == 'full'
    assert record['spans'] == ['entity', 'evidence1', 'evidence2', 'instruction']
    assert record['development_roles'] == ['fit', 'validation']
    assert record['full_pair_count'] == 4024
    assert record['development_pair_count'] == 3016
    assert panel.panel_sha256 == sha256_json(record)
    assert panel == LocalizationCandidatePanel(slug, count)
    assert len(layers) == len(set(layers)) and all(0 <= x < count for x in layers)


@pytest.mark.parametrize('slug', ['Qwen3.5-4B', 'glm-4-9b', 'gemma-4-12b',
                                  'openai/gpt-oss-20b', 'unknown', '', None, []])
def test_no_alias_or_unknown_model(slug):
    with pytest.raises(ValueError):
        LocalizationCandidatePanel(slug, 32)


@pytest.mark.parametrize('count', [True, False, 0, -1, 31, 33, 32.0, '32', None])
def test_actual_count_required_and_authenticated(count):
    with pytest.raises(ValueError):
        LocalizationCandidatePanel('qwen3.5-4b', count)



def test_all_models_reject_different_positive_counts():
    for slug, count, *_ in CASES:
        with pytest.raises(ValueError):
            LocalizationCandidatePanel(slug, count + 1)


def test_neighbor_clipping_and_anchor_deduplication():
    assert _candidate_layers(1, 0, 0) == (0,)
    assert _candidate_layers(4, 0, 0) == (0, 1, 3)
    assert _candidate_layers(4, 3, 3) == (0, 1, 2, 3)


def test_no_alternate_layers_source_outcome_or_sampling_arguments():
    for name in ('layers', 'spans', 'outcomes', 'scores', 'tickers', 'donors',
                 'evaluation', 'pair_count', 'historical_peak', 'source_path'):
        with pytest.raises(TypeError):
            build_localization_candidate_panel(model_slug='qwen3.5-4b',
                                               actual_layer_count=32, **{name: []})
    with pytest.raises(TypeError):
        build_localization_candidate_panel(model_slug='qwen3.5-4b')


def test_immutable_and_defensive_canonical_export():
    panel = LocalizationCandidatePanel('qwen3.5-4b', 32)
    digest = panel.panel_sha256
    with pytest.raises(FrozenInstanceError):
        panel.actual_layer_count = 40
    with pytest.raises(TypeError):
        panel.layers[0] = 1
    record = panel.to_dict()
    for value in record.values():
        if isinstance(value, list):
            value.clear()
    assert panel.panel_sha256 == digest
    with pytest.raises(ValueError):
        replace(panel, actual_layer_count=40)


def test_historical_file_binding_and_drift_rejection(monkeypatch):
    path = Path(__file__).resolve().parents[1] / HISTORICAL_SOURCE
    original = path.read_bytes()
    assert sha256(original).hexdigest() == HISTORICAL_SOURCE_SHA256
    assert all(s in original.decode() for s in ('L15', 'L20', 'L27', 'L8', 'L16', 'L19', 'L14'))
    monkeypatch.setattr(Path, 'read_bytes', lambda self: original + b'changed')
    with pytest.raises(ValueError, match='historical source'):
        LocalizationCandidatePanel('qwen3.5-4b', 32)

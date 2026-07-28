from llm_bias.counterfactual_patching.visualization import _top1_comparison


def _grid(rows):
    return {
        "seq_len": 2,
        "grid": [
            {
                "layer": layer,
                "top_ids": [[source, 0], [source + 1, 0]],
                "top_probs": [[0.6, 0.1], [0.7, 0.1]],
            }
            for layer, source in rows
        ],
    }


def test_top1_comparison_keeps_layer_and_position_alignment():
    source = _grid([(0, 10), (4, 20)])
    target = _grid([(0, 11), (4, 21)])
    patched = _grid([(0, 10), (4, 21)])

    result = _top1_comparison(source, target, patched)

    assert len(result) == 4
    assert result[0] == {
        "layer": 0,
        "position": 0,
        "source_top1": 10,
        "target_top1": 11,
        "patched_top1": 10,
        "source_top1_prob": 0.6,
        "target_top1_prob": 0.6,
        "patched_top1_prob": 0.6,
    }

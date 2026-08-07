from llm_bias.counterfactual_patching.binary_summary import summarize_baseline


def test_baseline_summary_pairs_option_orders_by_career():
    rows = [
        {"career_id": "a", "split": "confirmation", "prompt_order": "dad_first", "margin": -1.0},
        {"career_id": "a", "split": "confirmation", "prompt_order": "mom_first", "margin": 1.0},
        {"career_id": "b", "split": "confirmation", "prompt_order": "dad_first", "margin": -2.0},
        {"career_id": "b", "split": "confirmation", "prompt_order": "mom_first", "margin": 2.0},
    ]
    summary = summarize_baseline(rows, seed=3, n_resamples=100)
    paired = summary["splits"]["confirmation"]["paired_orders"]
    assert paired["pair_count"] == 2
    assert paired["order_effect_mom_minus_dad"] == 3.0
    assert paired["same_sign_ratio"] == 0.0

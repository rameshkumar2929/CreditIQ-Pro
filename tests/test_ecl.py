from creditiq.ecl import calculate_ecl

def test_ecl_weighted_sum():
    result = calculate_ecl(0.10, 0.40, 1000, 0.05, 1, [
        {"name":"base", "weight":0.5, "pd_multiplier":1.0},
        {"name":"stress", "weight":0.5, "pd_multiplier":2.0},
    ])
    assert result["weighted_discounted_ecl"] > 0
    assert len(result["scenarios"]) == 2

def test_ecl_requires_weights_to_sum_to_one():
    try:
        calculate_ecl(0.1, 0.4, 1000, 0.05, 1, [{"name":"base", "weight":0.8}])
    except ValueError:
        return
    assert False, "Expected invalid scenario weights to raise ValueError"

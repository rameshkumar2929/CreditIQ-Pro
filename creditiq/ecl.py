from __future__ import annotations

def calculate_ecl(pd_base: float, lgd: float, ead: float, eir: float, horizon_years: float, scenarios: list[dict]) -> dict:
    """Simple illustrative scenario-weighted PD×LGD×EAD with discounting."""
    for name, value in (("PD", pd_base), ("LGD", lgd), ("EIR", eir)):
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if float(ead) < 0 or float(horizon_years) < 0:
        raise ValueError("EAD and horizon must be non-negative.")
    weight_sum = sum(float(s["weight"]) for s in scenarios)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("Scenario weights must sum to 1.")
    rows = []
    for scenario in scenarios:
        weight = float(scenario["weight"])
        multiplier = float(scenario.get("pd_multiplier", 1.0))
        scenario_pd = min(max(float(pd_base) * multiplier, 0.0), 1.0)
        undiscounted = scenario_pd * float(lgd) * float(ead)
        discounted = undiscounted / ((1.0 + float(eir)) ** float(horizon_years))
        rows.append({
            "scenario": scenario["name"],
            "weight": weight,
            "scenario_pd": scenario_pd,
            "lgd": float(lgd),
            "ead": float(ead),
            "undiscounted_ecl": undiscounted,
            "discounted_ecl": discounted,
            "weighted_discounted_ecl": weight * discounted,
        })
    return {"scenarios": rows, "weighted_discounted_ecl": sum(r["weighted_discounted_ecl"] for r in rows)}

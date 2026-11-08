"""Checks for model.py. Run with ``python -m pytest experiments/russell_impulses``."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import model as m  # noqa: E402


def const(p):
    return lambda t: p


@pytest.mark.parametrize("p", [
    m.BASE,
    m.BASE.replace(r_p=1.5),
    m.BASE.replace(r_p=1.5, alpha=0.75),
    m.POLARIZED,
    m.POLARIZED.replace(r_p=1.5),
])
def test_equilibria_are_fixed_points(p):
    for e in m.equilibria(p):
        assert np.allclose(m.rhs(np.array([e["C"], e["P"]]), p), 0, atol=1e-12)


def test_regime_matches_stability():
    cases = {
        m.BASE: "stable coexistence",
        m.BASE.replace(r_p=1.5, alpha=0.75): "possessive excludes creative",
        m.POLARIZED: "bistable: the starting state decides",
        m.BASE.replace(beta=0.75): "creative excludes possessive",
    }
    for p, want in cases.items():
        assert m.regime(p) == want
        stable = {e["name"] for e in m.equilibria(p) if e["kind"] == "stable"}
        expect = {
            "stable coexistence": {"coexistence"},
            "possessive excludes creative": {"possessive only"},
            "creative excludes possessive": {"creative only"},
            "bistable: the starting state decides": {"creative only", "possessive only"},
        }[want]
        assert stable == expect


def test_rp_alone_never_excludes_creative():
    # C = 0 is reachable only when alpha > r_c, whatever r_p is.
    for r_p in [0.1, 0.5, 1.5, 5.0, 50.0]:
        p = m.BASE.replace(r_p=r_p)
        r = m.simulate(const(p), (0.6, 0.6), years=200)
        assert r["C"][-1] > 0.4


def test_rk4_converges():
    sc = m.scenarios(5, 9)[2]
    a = m.simulate(sc["schedule"], sc["y0"], dt=0.01)
    b = m.simulate(sc["schedule"], sc["y0"], dt=0.005)
    assert np.max(np.abs(a["C"] - b["C"])) < 1e-8
    assert np.max(np.abs(a["P"] - b["P"])) < 1e-8


def test_scenarios_and_bounds():
    s1, s2, s3, s4a, s4b = (m.simulate(sc["schedule"], sc["y0"]) for sc in m.scenarios(5, 9))
    for r in (s1, s2, s3, s4a, s4b):
        assert r["year"][0] == 0 and r["year"][-1] == m.YEARS and len(r["year"]) == m.YEARS + 1
        assert np.all(r["C"] >= 0) and np.all(r["P"] >= 0)
        assert np.all(r["C"] <= 1 + 1e-9) and np.all(r["P"] <= 1 + 1e-9)
    assert abs(s1["C"][-1] - 2 / 3) < 0.01            # recovers
    assert abs(s2["C"][-1] - 6 / 11) < 0.001          # new coexistence point
    assert s3["C"][-1] < 1e-3                         # excluded
    assert s4a["C"][-1] > s4a["P"][-1]                # short crisis: recovers
    assert s4b["P"][-1] > s4b["C"][-1]                # long crisis: tipped


def test_parameter_switch_timing():
    sc = m.scenarios(5, 9)[0]
    r = m.simulate(sc["schedule"], sc["y0"])
    # sample at year t records the parameters in force from t onward
    assert r["r_p"][9] == 0.5 and r["r_p"][10] == 1.5
    assert r["r_p"][19] == 1.5 and r["r_p"][20] == 0.5


def test_borderline_and_marginal_cases():
    p = m.BASE.replace(alpha=0.5)  # alpha == r_c
    assert m.regime(p).startswith("borderline")
    kinds = {e["name"]: e["kind"] for e in m.equilibria(p)}
    assert kinds["possessive only"] == "marginal"
    with pytest.raises(ValueError):
        m.Params(r_c=0, r_p=0.5, alpha=0.1, beta=0.1)


def test_partial_year_horizon_keeps_final_state():
    r = m.simulate(const(m.BASE), (0.1, 0.1), years=2.5)
    assert r["year"][-1] == 2.5


def test_critical_length_brackets():
    t = m.critical_crisis_length()
    assert 5 < t < 9
    assert m.polarized_outcome(t - 0.05) == "creative"
    assert m.polarized_outcome(t + 0.05) == "possessive"
    assert math.isfinite(t)

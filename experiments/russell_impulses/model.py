"""Creative vs. possessive impulses as a two-species competition model.

Bertrand Russell, *Principles of Social Reconstruction* (1916), separates
creative impulses (making things whose use does not deprive anyone else) from
possessive impulses (acquiring or defending things others cannot then have).
This module models the two as competing quantities on a 0..1 scale:

    dC/dt = r_c * C * (1 - C) - alpha * P * C
    dP/dt = r_p * P * (1 - P) - beta  * C * P

C  creative impulse, proxied by public spending on arts, culture, higher
   education, humanitarian aid and environmental protection
P  possessive impulse, proxied by spending on defence, rearmament, border
   security, corporate bailouts and domestic policing
r_c, r_p  intrinsic growth rates (per year)
alpha     how strongly P displaces C ("choking factor")
beta      how strongly C pushes back against P ("resistance factor")

Each variable is a fraction of its own saturation level (carrying capacity 1),
not a share of one shared budget, so C + P may exceed 1.

The parameters are illustrative. Nothing here is fitted to any country's data.

Run ``python model.py`` to write per-scenario CSVs, a summary and a figure to
``results/``. Only NumPy is required; the figure needs matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

YEARS = 50
DT = 0.01  # years per RK4 step; 1/DT must be an integer


@dataclass(frozen=True)
class Params:
    r_c: float
    r_p: float
    alpha: float
    beta: float

    def replace(self, **kw) -> "Params":
        d = self.__dict__.copy()
        d.update(kw)
        return Params(**d)


Schedule = Callable[[float], Params]


def rhs(y: np.ndarray, p: Params) -> np.ndarray:
    C, P = y
    return np.array([
        p.r_c * C * (1.0 - C) - p.alpha * P * C,
        p.r_p * P * (1.0 - P) - p.beta * C * P,
    ])


def simulate(schedule: Schedule, y0: tuple[float, float], years: float = YEARS,
             dt: float = DT, samples_per_year: int = 1) -> dict[str, np.ndarray]:
    """Integrate with classical RK4 and return ``samples_per_year`` samples a year.

    Parameters are held constant within each step and read at the step's
    midpoint, so a switch at a whole year never falls inside a step.
    """
    steps_per_year = round(1.0 / dt)
    if not math.isclose(steps_per_year * dt, 1.0):
        raise ValueError("1/dt must be an integer")
    if steps_per_year % samples_per_year:
        raise ValueError("samples_per_year must divide 1/dt")
    every = steps_per_year // samples_per_year
    n_steps = round(years * steps_per_year)
    y = np.array(y0, dtype=float)
    out_t, out_y, out_p = [0.0], [y.copy()], [schedule(0.0)]
    for n in range(n_steps):
        p = schedule((n + 0.5) * dt)
        k1 = rhs(y, p)
        k2 = rhs(y + 0.5 * dt * k1, p)
        k3 = rhs(y + 0.5 * dt * k2, p)
        k4 = rhs(y + dt * k3, p)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (n + 1) % every == 0:
            t = (n + 1) * dt
            out_t.append(round(t, 9))
            out_y.append(y.copy())
            out_p.append(p)
    ys = np.array(out_y)
    return {
        "year": np.array(out_t),
        "C": ys[:, 0],
        "P": ys[:, 1],
        "r_c": np.array([q.r_c for q in out_p]),
        "r_p": np.array([q.r_p for q in out_p]),
        "alpha": np.array([q.alpha for q in out_p]),
        "beta": np.array([q.beta for q in out_p]),
    }


# --------------------------------------------------------------------------
# Equilibria and their stability
# --------------------------------------------------------------------------

def jacobian(C: float, P: float, p: Params) -> np.ndarray:
    return np.array([
        [p.r_c * (1 - 2 * C) - p.alpha * P, -p.alpha * C],
        [-p.beta * P, p.r_p * (1 - 2 * P) - p.beta * C],
    ])


def equilibria(p: Params) -> list[dict]:
    """All equilibria in the closed unit square, with eigenvalues and type."""
    a, b = p.alpha / p.r_c, p.beta / p.r_p
    pts = [("extinct", 0.0, 0.0), ("creative only", 1.0, 0.0), ("possessive only", 0.0, 1.0)]
    if not math.isclose(a * b, 1.0):
        C = (1 - a) / (1 - a * b)
        P = (1 - b) / (1 - a * b)
        if C > 0 and P > 0:
            pts.append(("coexistence", C, P))
    res = []
    for name, C, P in pts:
        ev = np.linalg.eigvals(jacobian(C, P, p))
        re = ev.real
        if np.all(re < 0):
            kind = "stable"
        elif np.all(re > 0):
            kind = "unstable"
        else:
            kind = "saddle"
        res.append({"name": name, "C": C, "P": P, "eig": ev, "kind": kind})
    return res


def regime(p: Params) -> str:
    """Long-run outcome class from the two invasion conditions.

    C can grow into a P-only society iff alpha < r_c.
    P can grow into a C-only society iff beta < r_p.
    """
    c_invades = p.alpha < p.r_c
    p_invades = p.beta < p.r_p
    if c_invades and p_invades:
        return "stable coexistence"
    if c_invades:
        return "creative excludes possessive"
    if p_invades:
        return "possessive excludes creative"
    return "bistable: the starting state decides"


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

BASE = Params(r_c=0.5, r_p=0.5, alpha=0.25, beta=0.25)
CRISIS_START = 10
CRISIS_RP = 1.5

# Strong mutual competition: each impulse, once dominant, keeps the other out.
POLARIZED = Params(r_c=0.5, r_p=0.5, alpha=0.6, beta=0.6)
RESIDUAL_P = 0.02  # possessive spending that never quite goes away


def balanced_start(p: Params = BASE) -> tuple[float, float]:
    eq = next(e for e in equilibria(p) if e["name"] == "coexistence")
    return (eq["C"], eq["P"])


def window(base: Params, start: float, end: float, **crisis) -> Schedule:
    """Base parameters, with ``crisis`` overrides on [start, end)."""
    hot = base.replace(**crisis)
    return lambda t: hot if start <= t < end else base


def scenarios(t_crisis_short: float, t_crisis_long: float) -> list[dict]:
    end = CRISIS_START + 10
    return [
        {
            "key": "s1_crisis_passes",
            "title": "1. Crisis that passes",
            "desc": f"r_p rises 0.5 -> {CRISIS_RP} for years {CRISIS_START}-{end}, then returns",
            "schedule": window(BASE, CRISIS_START, end, r_p=CRISIS_RP),
            "y0": balanced_start(),
        },
        {
            "key": "s2_crisis_permanent",
            "title": "2. Crisis budget made permanent",
            "desc": f"r_p rises 0.5 -> {CRISIS_RP} at year {CRISIS_START} and stays",
            "schedule": window(BASE, CRISIS_START, math.inf, r_p=CRISIS_RP),
            "y0": balanced_start(),
        },
        {
            "key": "s3_displacement",
            "title": "3. Displacement institutionalized",
            "desc": f"as 2, and alpha rises 0.25 -> 0.75 (> r_c) at year {CRISIS_START}",
            "schedule": window(BASE, CRISIS_START, math.inf, r_p=CRISIS_RP, alpha=0.75),
            "y0": balanced_start(),
        },
        {
            "key": "s4a_polarized_short",
            "title": f"4a. Polarized society, {t_crisis_short:g}-year crisis",
            "desc": f"alpha = beta = 0.6; r_p = {CRISIS_RP} for {t_crisis_short:g} years from year {CRISIS_START}",
            "schedule": window(POLARIZED, CRISIS_START, CRISIS_START + t_crisis_short, r_p=CRISIS_RP),
            "y0": (1.0, RESIDUAL_P),
        },
        {
            "key": "s4b_polarized_long",
            "title": f"4b. Polarized society, {t_crisis_long:g}-year crisis",
            "desc": f"alpha = beta = 0.6; r_p = {CRISIS_RP} for {t_crisis_long:g} years from year {CRISIS_START}",
            "schedule": window(POLARIZED, CRISIS_START, CRISIS_START + t_crisis_long, r_p=CRISIS_RP),
            "y0": (1.0, RESIDUAL_P),
        },
    ]


def polarized_outcome(crisis_years: float, horizon: float = 200.0) -> str:
    """Which absorbing state the polarized society reaches after a crisis."""
    sched = window(POLARIZED, CRISIS_START, CRISIS_START + crisis_years, r_p=CRISIS_RP)
    r = simulate(sched, (1.0, RESIDUAL_P), years=CRISIS_START + crisis_years + horizon)
    return "possessive" if r["P"][-1] > r["C"][-1] else "creative"


def critical_crisis_length(lo: float = 0.0, hi: float = 30.0, tol: float = 0.01) -> float:
    """Shortest crisis (years) after which the polarized society never recovers.

    Crisis lengths are resolved to the integrator step, so the result is
    accurate to about DT.
    """
    if polarized_outcome(lo) != "creative" or polarized_outcome(hi) != "possessive":
        raise RuntimeError("bracket does not straddle the tipping point")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if polarized_outcome(mid) == "possessive":
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

COLS = ["year", "C", "P", "r_c", "r_p", "alpha", "beta"]


def write_csv(path: Path, r: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i in range(len(r["year"])):
            w.writerow([f"{r['year'][i]:g}"] + [f"{r[c][i]:.6f}" for c in COLS[1:]])


def fmt_eq(p: Params) -> list[str]:
    lines = [f"  r_c={p.r_c:g} r_p={p.r_p:g} alpha={p.alpha:g} beta={p.beta:g}"
             f"  (alpha/r_c={p.alpha / p.r_c:.3g}, beta/r_p={p.beta / p.r_p:.3g}) -> {regime(p)}"]
    for e in equilibria(p):
        ev = ", ".join(f"{z.real:+.3f}" if abs(z.imag) < 1e-12 else f"{z:.3f}" for z in e["eig"])
        lines.append(f"    {e['name']:<16} C={e['C']:.3f} P={e['P']:.3f}  eig=[{ev}]  {e['kind']}")
    return lines


def plot(runs: list[tuple[dict, dict]], t_crit: float, path: Path) -> None:
    """``runs`` pairs each scenario with a finely sampled trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    c_col, p_col, band = "#2a78d6", "#eb6834", "#f1f0ec"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": ink2,
        "xtick.color": ink2, "ytick.color": ink2, "text.color": ink,
    })
    fig, axes = plt.subplots(len(runs), 1, figsize=(7.5, 2.1 * len(runs)), sharex=True,
                             facecolor=surface)
    for ax, (sc, r) in zip(axes, runs):
        ax.set_facecolor(surface)
        crisis = r["r_p"] > r["r_p"][0] + 1e-12
        if crisis.any():
            step = r["year"][1] - r["year"][0]
            yrs = r["year"][crisis]
            ax.axvspan(yrs[0] - step, yrs[-1], color=band, lw=0, zorder=0)
        ax.plot(r["year"], r["C"], color=c_col, lw=2, label="Creative (C)")
        ax.plot(r["year"], r["P"], color=p_col, lw=2, label="Possessive (P)")
        c_end, p_end = r["C"][-1], r["P"][-1]
        gap = 0.09  # keep the two end labels from overlapping
        c_y, p_y = c_end, p_end
        if abs(c_y - p_y) < gap:
            mid = 0.5 * (c_y + p_y)
            c_y, p_y = (mid - gap / 2, mid + gap / 2) if c_end <= p_end else (mid + gap / 2, mid - gap / 2)
        ax.text(r["year"][-1] + 0.6, c_y, f"C {c_end:.2f}", color=ink2, va="center")
        ax.text(r["year"][-1] + 0.6, p_y, f"P {p_end:.2f}", color=ink2, va="center")
        ax.set_ylim(-0.03, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.set_title(sc["title"], loc="left", fontsize=10, color=ink)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.32))
    axes[-1].set_xlabel("year")
    axes[-1].set_xlim(0, YEARS + 6)
    fig.suptitle("Creative vs. possessive impulses, 50 years (shaded: crisis parameters in force)", x=0.02,
                 ha="left", fontsize=11)
    fig.text(0.02, 0.005, f"Polarized society tips permanently once the crisis lasts "
             f"more than {t_crit:.2f} years. Illustrative parameters, not fitted to data.",
             color=ink2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    t_crit = critical_crisis_length()
    short, long_ = math.floor(t_crit * 0.75), math.ceil(t_crit * 1.25)
    runs = []
    lines = ["Equilibria (C, P) and Jacobian eigenvalues", ""]
    for label, p in [("Baseline", BASE),
                     (f"Crisis (r_p={CRISIS_RP})", BASE.replace(r_p=CRISIS_RP)),
                     ("Crisis + displacement (alpha=0.75)", BASE.replace(r_p=CRISIS_RP, alpha=0.75)),
                     ("Polarized peacetime", POLARIZED),
                     (f"Polarized crisis (r_p={CRISIS_RP})", POLARIZED.replace(r_p=CRISIS_RP))]:
        lines.append(label)
        lines += fmt_eq(p)
        lines.append("")
    lines.append(f"Critical crisis length for the polarized society: {t_crit:.2f} years")
    lines.append("")
    lines.append("Trajectories (C / P)")
    sample = [0, 10, 15, 20, 30, 40, 50]
    lines.append(f"{'scenario':<38}" + "".join(f"{'y' + str(y):>13}" for y in sample))
    for sc in scenarios(short, long_):
        r = simulate(sc["schedule"], sc["y0"])
        write_csv(args.out / f"{sc['key']}.csv", r)
        runs.append((sc, r))
        cells = "".join(f"{r['C'][y]:>6.3f}/{r['P'][y]:<6.3f}" for y in sample)
        lines.append(f"{sc['title']:<38}{cells}")
    lines.append("")
    for sc, _ in runs:
        lines.append(f"{sc['title']}: {sc['desc']}")
    text = "\n".join(lines) + "\n"
    (args.out / "summary.txt").write_text(text)
    print(text)
    if not args.no_plot:
        fine = [(sc, simulate(sc["schedule"], sc["y0"], samples_per_year=20)) for sc, _ in runs]
        plot(fine, t_crit, args.out / "trajectories.png")
        print(f"figure: {args.out / 'trajectories.png'}")


if __name__ == "__main__":
    main()

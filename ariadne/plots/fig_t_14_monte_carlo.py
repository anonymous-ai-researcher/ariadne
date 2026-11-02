"""Figure T14: the closed form against Monte Carlo.

Reads     D1_closed_form/mc.json and mc_seed*.json  (rows over k, D, alpha = D_s / D and c: "closed", "mc", "se", "z",
          "low" (too few simulated conflicts for a z-score), and the same for the rank-1 share and for the
          conflict probability given rank 1: "rank1_*", "cond1_*")
Shows     (a) simulated conflict probability against the closed form for every cell of the grid and every seed,
              on log scales; shade = tilt alpha (dark: siblings concentrated at the query), hollow = cells with
              too few simulated conflicts for a test.
          (b) the same comparison as z-scores (simulated minus closed form, in standard errors) against the
              closed-form value: no trend, and the band of +/- 2 holds about 95% of the cells.
          (c) distribution of the z-scores of the three tested quantities (conflict probability, rank-1 share,
              conflict given rank 1) against the standard normal density.
Paper     appendix; one sentence in the main text (largest |z|, share beyond 2).

Options   none
"""
import numpy as np

from . import _theory as T

NAME = "fig_t_14_monte_carlo"


def main(results="results", suffix="", out="figures"):
    plt = T.setup()
    from matplotlib.colors import to_hex
    files = T.jglob(results, "D1_closed_form", "mc*.json", suffix)
    rows = [r for f in files.values() for r in (f.get("rows") or [])]
    if not rows:
        return T.skip(NAME, "no mc*.json with rows in %s" % T.exp_dir(results, "D1_closed_form", suffix))
    alphas = sorted({float(r["alpha"]) for r in rows})
    acol = {a: to_hex(T.COOL_CMAP(0.98 - 0.62 * (i / max(len(alphas) - 1, 1)))) for i, a in enumerate(alphas)}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(T.COL2, 2.35), gridspec_kw=dict(width_ratios=[1.0, 1.25, 0.9]))
    cl = np.array([T.fnum(r["closed"]) for r in rows]); mc = np.array([T.fnum(r["mc"]) for r in rows])
    z = np.array([T.fnum(r.get("z")) for r in rows]); low = np.array([bool(r.get("low")) for r in rows])
    al = np.array([float(r["alpha"]) for r in rows])
    pos = (cl > 0) & (mc > 0)
    lo = float(min(cl[pos].min(), mc[pos].min())) / 2; hi = 1.6
    a1.plot([lo, hi], [lo, hi], color=T.INK, lw=0.6, zorder=1)
    for a in alphas:
        s = pos & (al == a)
        a1.plot(cl[s & ~low], mc[s & ~low], ls="none", marker="o", ms=2.6, mfc=acol[a], mec="none", alpha=0.85, zorder=3)
        a1.plot(cl[s & low], mc[s & low], ls="none", marker="o", ms=2.6, mfc="none", mec=acol[a], mew=0.5, zorder=2)
        t = ~low & (al == a) & np.isfinite(z) & (cl > 0)
        a2.plot(cl[t], z[t], ls="none", marker="o", ms=2.6, mfc=acol[a], mec="none", alpha=0.85, zorder=3)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lo, hi); a1.set_ylim(lo, hi); a1.set_aspect("equal", adjustable="box")
    a1.set_xlabel("closed form"); a1.set_ylabel("Monte Carlo")
    a1.grid(True, color=T.LIGHT, lw=0.4)
    for j, a in enumerate(alphas):
        a1.plot([0.07], [0.80 - 0.085 * j], transform=a1.transAxes, ls="none", marker="o", ms=3.0, mfc=acol[a], mec="none")
        a1.text(0.13, 0.80 - 0.085 * j, "%g" % a, transform=a1.transAxes, fontsize=T.TINY, ha="left", va="center")
    a1.text(0.04, 0.955, "$\\alpha = D_s/D$", transform=a1.transAxes, fontsize=T.TINY, ha="left", va="top", color=T.MUTED)
    if low.any():                                                       # key of the hollow markers, in the empty lower-right corner
        a1.plot([0.47], [0.115], transform=a1.transAxes, ls="none", marker="o", ms=2.6, mfc="none", mec=T.MUTED, mew=0.5, zorder=6)
        a1.text(0.52, 0.115, "too few simulated\nconflicts (not tested)", transform=a1.transAxes, fontsize=T.TINY, ha="left", va="center",
                linespacing=1.1, zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    T.panel(a1, "a", "conflict probability")

    tested = ~low & np.isfinite(z)
    a2.axhspan(-2, 2, color="#EDEDED", lw=0, zorder=0); a2.axhline(0, color=T.INK, lw=0.6, zorder=1)
    a2.set_xscale("log"); a2.set_xlabel("closed form"); a2.set_ylabel("$z$ = (Monte Carlo $-$ closed form) / s.e.")
    zmax = float(np.nanmax(np.abs(z[tested]))) if tested.any() else 3.0
    a2.set_ylim(-max(4.2, zmax * 1.15), max(4.2, zmax * 1.15))
    a2.text(0.03, 0.96, "%d (setting, seed) pairs tested: max $|z|$ = %.2f, $|z| > 2$ in %.1f%%" % (int(tested.sum()), zmax, 100 * float(np.mean(np.abs(z[tested]) > 2)) if tested.any() else 0),
            transform=a2.transAxes, fontsize=T.TINY, ha="left", va="top")
    T.panel(a2, "b", "no trend in the residuals")

    xs = np.linspace(-4.5, 4.5, 200); bins = np.linspace(-4.5, 4.5, 28)
    sets = [("conflict", z[tested], T.INK)]
    for key, lab, col in (("rank1_z", "rank-1 share", T.C_SEMI1), ("cond1_z", "conflict | rank 1", T.C_CLOSED)):
        v = np.array([T.fnum(r.get(key)) for r in rows]); lowk = np.array([bool(r.get(key.replace("_z", "_low"), False)) for r in rows])
        sets.append((lab, v[np.isfinite(v) & ~lowk], col))
    for j, (lab, v, col) in enumerate(sets):
        if len(v) < 5:
            continue
        h, e = np.histogram(np.clip(v, -4.49, 4.49), bins, density=True)
        a3.plot(np.repeat(e, 2)[1:-1], np.repeat(h, 2), color=col, lw=0.9, zorder=3 + j, solid_joinstyle="miter")
        a3.text(0.97, 0.95 - 0.085 * j, "%s (%d)" % (lab, len(v)), transform=a3.transAxes, fontsize=T.TINY, ha="right", va="top", color=col)
    a3.fill_between(xs, 0, np.exp(-xs * xs / 2) / np.sqrt(2 * np.pi), color="#E2E2E2", lw=0, zorder=1)
    a3.set_xlim(-4.5, 4.5); a3.set_ylim(0, a3.get_ylim()[1] * 1.38); a3.set_yticks([]); a3.spines["left"].set_visible(False)
    a3.set_xlabel("$z$")
    T.panel(a3, "c", "$z$ against N(0, 1)")
    print("  %s: %d cells (%d files), %d tested: max |z| %.2f, share |z| > 2 %.3f" % (NAME, len(rows), len(files), int(tested.sum()), zmax,
                                                                                    float(np.mean(np.abs(z[tested]) > 2)) if tested.any() else float("nan")))
    return T.save(fig, NAME, out)


if __name__ == "__main__":
    T.cli(main, __doc__)

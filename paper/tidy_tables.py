#!/usr/bin/env python3
"""Typesetting of the generated tables (paper side; the generator in ariadne/plots/tables.py is not touched):
  * an interval printed in e-notation (floating-point residue of identical seeds, or below 0.001) is printed as 0 or 0.000;
  * "at most 1 rounds" -> "at most 1 round";
  * MS MARCO passages have no tag labels: its tag column reads n/a instead of 0 (tab_cost_all);
  * a table longer than MAXROWS body rows is split at its block boundaries into tables/<name>.tex and
    tables/<name>__2.tex, __3.tex, ... (the full version prints the parts as one table continued over pages).
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
MAXROWS = 62


def enot(m):
    v = float(m.group(1))
    return "$\\pm$ 0" if v < 1e-9 else "$\\pm$ 0.000"


def pm(m):
    """an interval is printed to the precision of its mean; an interval that rounds to zero is not printed (as for
    identical seeds)"""
    mean, dec, h = m.group(1), m.group(2), float(m.group(3))
    d = len(dec) if dec else 0
    hr = round(h, d)
    if hr == 0:
        return mean
    return "%s $\\pm$ %.*f" % (mean, d, hr)


def split(name, text):
    lines = text.split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.strip() == "\\midrule")           # end of the header
    i1 = max(i for i, l in enumerate(lines) if l.strip().startswith("\\bottomrule"))
    pre = [l for l in lines[:i0 + 1]]
    head = [l for l in pre if not l.startswith("%")]
    starts = [i for i in range(i0 + 1, i1) if lines[i].startswith("\\multicolumn") and lines[i - 1].strip() == "\\midrule"]
    if len(starts) < 2:
        return [text]
    blocks = []
    for a, b in zip(starts, starts[1:] + [i1 + 1]):
        blk = lines[a:b]
        while blk and blk[-1].strip() in ("\\midrule", "\\bottomrule", ""):
            blk.pop()
        blocks.append(blk)
    parts, cur = [], []
    for blk in blocks:
        if cur and sum(len(b) for b in cur) + len(blk) > MAXROWS:
            parts.append(cur); cur = []
        cur.append(blk)
    parts.append(cur)
    out = []
    for k, part in enumerate(parts):
        body = []
        for j, blk in enumerate(part):
            if j:
                body.append("\\midrule")
            body += blk
        out.append("\n".join((pre if k == 0 else head) + body + ["\\bottomrule", "\\end{tabular}", ""]))
    return out


def main():
    for p in sorted(glob.glob(os.path.join(TAB, "*.tex"))):
        if "__" in os.path.basename(p):
            os.remove(p)
    for p in sorted(glob.glob(os.path.join(TAB, "*.tex"))):
        name = os.path.basename(p)[:-4]
        t = open(p, encoding="utf-8").read()
        t = re.sub(r"\$\\pm\$ ([0-9.]+e-[0-9]+)", enot, t)
        t = re.sub(r"(-?[0-9][0-9,]*(?:\.([0-9]+))?) \$\\pm\$ ([0-9.]+)", pm, t)
        t = t.replace("at most 1 rounds", "at most 1 round")
        if name == "tab_cost_all":
            ls = t.split("\n"); ms = False
            for i, l in enumerate(ls):
                if l.startswith("\\multicolumn"):
                    ms = "MS MARCO" in l
                elif ms and "&" in l and not l.startswith("\\"):
                    c = l.split(" & ")
                    if len(c) >= 5 and c[4].strip() == "0":
                        c[4] = "n/a"; ls[i] = " & ".join(c)
            t = "\n".join(ls)
        body = sum(1 for l in t.split("\n") if "&" in l and not l.startswith("%"))
        parts = split(name, t) if body > MAXROWS else [t]
        open(p, "w", encoding="utf-8").write(parts[0])
        for k, q in enumerate(parts[1:], 2):
            open(os.path.join(TAB, "%s__%d.tex" % (name, k)), "w", encoding="utf-8").write(q)
        if len(parts) > 1:
            print("tidy_tables: %s split into %d parts" % (name, len(parts)))


if __name__ == "__main__":
    main()

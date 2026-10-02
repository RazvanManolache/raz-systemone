"""Offline per-question calibration (shreyanbr-style, fit on dumps).

Fit on a calib dump (unseen-by-train states!):
  raz eval --data tests/data/calib.jsonl --scorer nli --nli-model <dir> --dump calib-vN.jsonl
  training/.venv/Scripts/python.exe training/calibrate.py --fit calib-vN.jsonl --out calib-vN.json

Then apply to a *different* dump to measure honest transfer:
  training/.venv/Scripts/python.exe training/calibrate.py --apply calib-vN.json --dump holdout-C-vN.jsonl

Fits: softmax temperature per choice/score question (golden-section on NLL),
2-param Platt (logistic regression) per noul question. Logits are recovered
as log(probabilities), which is exact up to the additive constant softmax
ignores.
"""

import argparse
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def golden(f, lo=0.05, hi=20.0, iters=60):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    return (a + b) / 2


def ece(conf, correct, bins=10):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def parse(dump_path):
    """-> {qname: {'kind': choice|score|noul, 'logits': [[..]], 'gold': [i], 'names': [..]}}"""
    groups = {}
    for line in open(dump_path):
        j = json.loads(line)
        q, a, exp = j["question"], j["answer"], j["expected"]
        t = a["type"]
        if t == "noul":
            p = min(max(a["noul"], EPS), 1 - EPS)
            g = groups.setdefault(q, {"kind": "noul", "logits": [], "gold": []})
            g["logits"].append([math.log(p / (1 - p))])
            g["gold"].append(1 if exp else 0)
        else:
            probs = a["probabilities"]
            names = sorted(probs, key=str)
            g = groups.setdefault(q, {"kind": t, "logits": [], "gold": [], "names": names})
            assert g["names"] == names, f"option order drift in {q}"
            g["logits"].append([math.log(max(probs[k], EPS)) for k in names])
            gold = str(exp) if t == "score" else exp
            g["gold"].append(names.index(str(gold) if t == "score" else gold))
    return groups


def fit(groups):
    calib = {"temperatures": {}, "platt": {}}
    for q, g in groups.items():
        z = np.array(g["logits"])
        y = np.array(g["gold"])
        if g["kind"] == "noul":
            lr = LogisticRegression().fit(z, y)
            calib["platt"][q] = {"w": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        else:
            def nll(T, z=z, y=y):
                p = softmax(z / T)
                return float(-np.log(p[np.arange(len(y)), y] + EPS).mean())

            calib["temperatures"][q] = {"T": golden(nll), "nll_before": nll(1.0)}
    return calib


def report(groups, calib=None):
    tot_ok = tot_n = 0
    conf, corr = [], []
    nll = 0.0
    for q, g in groups.items():
        z = np.array(g["logits"])
        y = np.array(g["gold"])
        if g["kind"] == "noul":
            if calib and q in calib["platt"]:
                w, b = calib["platt"][q]["w"], calib["platt"][q]["b"]
                p1 = 1 / (1 + np.exp(-(w * z[:, 0] + b)))
            else:
                p1 = 1 / (1 + np.exp(-z[:, 0]))
            p = np.stack([1 - p1, p1], 1)
        else:
            T = calib["temperatures"][q]["T"] if calib and q in calib["temperatures"] else 1.0
            p = softmax(z / T)
        pred = p.argmax(1)
        tot_ok += int((pred == y).sum())
        tot_n += len(y)
        conf.extend(p.max(1).tolist())
        corr.extend((pred == y).tolist())
        nll += float(-np.log(p[np.arange(len(y)), y] + EPS).sum())
    return {"acc": tot_ok / tot_n, "nll": nll / tot_n, "ece": ece(conf, corr), "n": tot_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", help="dump to fit on -> needs --out")
    ap.add_argument("--out")
    ap.add_argument("--apply", help="calib.json to apply -> needs --dump")
    ap.add_argument("--dump")
    args = ap.parse_args()
    if args.fit:
        assert args.out, "--fit needs --out"
        groups = parse(args.fit)
        calib = fit(groups)
        json.dump(calib, open(args.out, "w"), indent=2)
        print("in-sample:", json.dumps(report(groups)))
        print("in-sample calibrated:", json.dumps(report(groups, calib)))
        print("wrote", args.out)
    elif args.apply:
        assert args.dump, "--apply needs --dump"
        calib = json.load(open(args.apply))
        groups = parse(args.dump)
        print("raw:", json.dumps(report(groups)))
        print("calibrated:", json.dumps(report(groups, calib)))


if __name__ == "__main__":
    main()

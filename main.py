import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from mpp import aliphatic_index, instability_index
HERE = Path(__file__).resolve().parent
WINDOW = 5
LOW_MARGIN = 0.10   # display-only: flag classifier probabilities this close to the threshold

KELCH_SEQ = "MEGEKVKTKANSISNFSMTYDRESGGNSNSDDKSGSSSENDSNSFMNLTSDKNEKTENNSFLLNNSSYGNVKDSLLESIDMSVLDSNFDSKKDFLPSNLSRTFNNMSKDNIGNKYLNKLLNKKKDTITNENNNINHNNNNNNLTANNITNNLINNNMNSPSIMNTNKKENFLDAANLINDDSGLNNLKKFSTVNNVNDTYEKKIIETELSDASDFENMVGDLRITFINWLKKTQMNFIREKDKLFKDKKELEMERVRLYKELENRKNIEEQKLHDERKKLDIDISNGYKQIKKEKEEHRKRFDEERLRFLQEIDKIKLVLYLEKEKYYQEYKNFENDKKKIVDANIATETMIDINVGGAIFETSRHTLTQQKDSFIEKLLSGRHHVTRDKQGRIFLDRDSELFRIILNFLRNPLTIPIPKDLSESEALLKEAEFYGIKFLPFPLVFCIGGFDGVEYLNSMELLDISQQCWRMCTPMSTKKAYFGSAVLNNFLYVFGGNNYDYKALFETEVYDRLRDVWYVSSNLNIPRRNNCGVTSNGRIYCIGGYDGSSIIPNVEAYDHRMKAWVEVAPLNTPRSSAMCVAFDNKIYVIGGTNGERLNSIEVYEEKMNKWEQFPYALLEARSSGAAFNYLNQIYVVGGIDNEHNILDSVEQYQPFNKRWQFLNGVPEKKMNFGAATLSDSYIITGGENGEVLNSCHFFSPDTNEWQLGPSLLVPRFGHSVLIANI"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

THREE_LETTER = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL',
}


#input handling and validation 
def parse_mutation(text):
    m = re.fullmatch(r"\s*([A-Za-z])(\d+)([A-Za-z])\s*", text)
    if not m:
        raise ValueError("Format must be <WT aa><position><mutant aa>, e.g. C580Y")
    og, resno, mut = m.group(1).upper(), int(m.group(2)), m.group(3).upper()

    if og not in VALID_AA or mut not in VALID_AA:
        raise ValueError("Use standard one-letter amino-acid codes.")
    if not 1 <= resno <= 726:
        raise ValueError("Position must be between 1 and 726")
    if KELCH_SEQ[resno - 1] != og:
        raise ValueError(
            f"Wrong input: residue {resno} in the reference sequence is "f"{KELCH_SEQ[resno - 1]}, not {og}."
        )
    if og == mut:
        raise ValueError("Mutant residue is the same as the WT residue.")
    return og, resno, mut


def build_mutant(resno, mut):
    return KELCH_SEQ[:resno - 1] + mut + KELCH_SEQ[resno:]


#importing and using mpp ALIN and II
def window_starts(resno, seq_len):
    """Start indices of all 5-residue windows that contain the mutated residue
    (0-based start index i covers residues i .. i+4). Clipped at the sequence ends."""
    first = max(0, resno - WINDOW)
    last = min(resno - 1, seq_len - WINDOW)
    return range(first, last + 1)


def mean_window_deviation(wt_seq, mut_seq, resno, func):
    devs = []
    for i in window_starts(resno, len(wt_seq)):
        wt_win = wt_seq[i:i + WINDOW]
        mut_win = mut_seq[i:i + WINDOW]
        devs.append(abs(func(mut_win) - func(wt_win)))   # window k of mutant vs window k of WT
    return sum(devs) / len(devs)


#calculating BDE values 
_bde_lookup = None


def load_bde_table():
    global _bde_lookup
    if _bde_lookup is None:
        d = pd.read_csv(HERE / "dipeptide.csv", skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        d["Model"] = d["Model"].str.strip()
        _bde_lookup = dict(zip(d["Model"], d["Ref. (kcal/mol)"]))
    return _bde_lookup


def get_bde(aa1, aa2, conformer=1):
    table = load_bde_table()
    fwd = f"{THREE_LETTER[aa1]}_{THREE_LETTER[aa2]}_{conformer}"
    rev = f"{THREE_LETTER[aa2]}_{THREE_LETTER[aa1]}_{conformer}"
    if fwd in table:
        return table[fwd]
    if rev in table:
        return table[rev]
    raise KeyError(f"Neither {fwd} nor {rev} found in dipeptide.csv")


def bde_deviation(og, mut, resno):
    devs = []
    if resno - 2 >= 0:
        left = KELCH_SEQ[resno - 2]
        devs.append(abs(get_bde(left, mut) - get_bde(left, og)))
    if resno < 726:               # right neighbour exists
        right = KELCH_SEQ[resno]
        devs.append(abs(get_bde(mut, right) - get_bde(og, right)))
    return sum(devs) / len(devs)


#final feature calculations
def compute_features(mutation):
    og, resno, mut = parse_mutation(mutation)
    mutant_seq = build_mutant(resno, mut)
    return {
        "II": mean_window_deviation(KELCH_SEQ, mutant_seq, resno, instability_index),
        "AI": mean_window_deviation(KELCH_SEQ, mutant_seq, resno, aliphatic_index),
        "BDE": bde_deviation(og, mut, resno),
    }


#loading model config
def load_config(name):
    with open(HERE / name) as f:
        return json.load(f)


def linear_score(cfg, feats):
    missing = [f for f in cfg["features"] if f not in feats]
    if missing:
        raise KeyError(f"Model needs feature(s) this tool does not compute: {missing}")
    return cfg["intercept"] + sum(
        c * (feats[f] - m) / s
        for f, c, m, s in zip(cfg["features"], cfg["coef"], cfg["mean"], cfg["scale"])
    )


def predict_resistance(cfg, feats):
    z = linear_score(cfg, feats)
    p = 1.0 / (1.0 + math.exp(-z))
    return ("Resistant" if p >= cfg["threshold"] else "Sensitive"), p


def predict_half_life(cfg, feats):
    pred_log = linear_score(cfg, feats)
    point = math.exp(pred_log)
    interval = None
    pi = cfg.get("pi")
    if pi and len(cfg["features"]) == 1:
        x = feats[cfg["features"][0]]
        se = pi["s"] * math.sqrt(1 + 1 / pi["n"] + (x - pi["x_mean"]) ** 2 / pi["sxx"])
        interval = (math.exp(pred_log - pi["t_crit"] * se),
                    math.exp(pred_log + pi["t_crit"] * se))
    return point, interval

#execution block
def main():
    text = sys.argv[1] if len(sys.argv) > 1 else input(
        "Enter mutation (format: <WT AA><residue number><mutant AA>, e.g. C580Y): ")
    try:
        feats = compute_features(text)
    except (ValueError, KeyError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    clf_cfg = load_config("classifier_config.json")
    reg_cfg = load_config("regression_config.json")
    label, prob = predict_resistance(clf_cfg, feats)
    half_life, interval = predict_half_life(reg_cfg, feats)

    print(f"\nMutation: {text.strip().upper()}")
    print("Features (deviation from WT): " +
          ", ".join(f"{k}={v:.3f}" for k, v in feats.items()))

    print("\n--- Resistance classification ---")
    print(f"Prediction     : {label}")
    print(f"P(Resistant)   : {prob:.3f}   (threshold {clf_cfg['threshold']})")
    if abs(prob - clf_cfg["threshold"]) < LOW_MARGIN:
        print("Note           : probability is close to the threshold; low-confidence call.")

    print("\n--- Clearance half-life (PC1/2) ---")
    print(f"Predicted PC1/2: {half_life:.3f}")
    if interval:
        print(f"95% prediction interval: {interval[0]:.3f} - {interval[1]:.3f}")
    print("Note           : regression has weak predictive power (LOOCV Q2 ~0.1); "
          "treat as a rough estimate.")

    print(f"\nModels: classifier [{clf_cfg.get('dataset', 'n/a')}], "
          f"regression [{reg_cfg.get('dataset', 'n/a')}]")
    print("Research tool, not a clinical diagnostic.")


if __name__ == "__main__":
    main()
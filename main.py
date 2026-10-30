import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
from mpp import aliphatic_index, instability_index

HERE = Path(__file__).resolve().parent
WINDOW = 5
LOW_MARGIN = 0.10   #close calls

KELCH_SEQ = "MEGEKVKTKANSISNFSMTYDRESGGNSNSDDKSGSSSENDSNSFMNLTSDKNEKTENNSFLLNNSSYGNVKDSLLESIDMSVLDSNFDSKKDFLPSNLSRTFNNMSKDNIGNKYLNKLLNKKKDTITNENNNINHNNNNNNLTANNITNNLINNNMNSPSIMNTNKKENFLDAANLINDDSGLNNLKKFSTVNNVNDTYEKKIIETELSDASDFENMVGDLRITFINWLKKTQMNFIREKDKLFKDKKELEMERVRLYKELENRKNIEEQKLHDERKKLDIDISNGYKQIKKEKEEHRKRFDEERLRFLQEIDKIKLVLYLEKEKYYQEYKNFENDKKKIVDANIATETMIDINVGGAIFETSRHTLTQQKDSFIEKLLSGRHHVTRDKQGRIFLDRDSELFRIILNFLRNPLTIPIPKDLSESEALLKEAEFYGIKFLPFPLVFCIGGFDGVEYLNSMELLDISQQCWRMCTPMSTKKAYFGSAVLNNFLYVFGGNNYDYKALFETEVYDRLRDVWYVSSNLNIPRRNNCGVTSNGRIYCIGGYDGSSIIPNVEAYDHRMKAWVEVAPLNTPRSSAMCVAFDNKIYVIGGTNGERLNSIEVYEEKMNKWEQFPYALLEARSSGAAFNYLNQIYVVGGIDNEHNILDSVEQYQPFNKRWQFLNGVPEKKMNFGAATLSDSYIITGGENGEVLNSCHFFSPDTNEWQLGPSLLVPRFGHSVLIANI"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

THREE_LETTER = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
    'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
    'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
    'Y': 'TYR', 'V': 'VAL',
}


#input handling
def parse_mutation(text):
    m = re.fullmatch(r"\s*([A-Za-z])(\d+)([A-Za-z])\s*", text)
    if not m:
        raise ValueError("Format must be <WT aa><position><mutant aa>, e.g. C580Y")
    og, resno, mut = m.group(1).upper(), int(m.group(2)), m.group(3).upper()

    if og not in VALID_AA or mut not in VALID_AA:
        raise ValueError("Use standard one-letter amino-acid codes.")
    if not 1 <= resno <= len(KELCH_SEQ):
        raise ValueError(f"Position must be between 1 and {len(KELCH_SEQ)}.")
    if KELCH_SEQ[resno - 1] != og:
        raise ValueError(
            f"Wrong input: residue {resno} in the reference sequence is "
            f"{KELCH_SEQ[resno - 1]}, not {og}."
        )
    if og == mut:
        raise ValueError("Mutant residue is the same as the WT residue.")
    return og, resno, mut

#build mutant sequence
def build_mutant(resno, mut):
    return KELCH_SEQ[:resno - 1] + mut + KELCH_SEQ[resno:]

#sequence sectioning into windows of 5
def window_starts(resno, seq_len):
    first = max(0, resno - WINDOW)
    last = min(resno - 1, seq_len - WINDOW)
    return range(first, last + 1)

#calculating mean ALIN and II
def mean_window_deviation(wt_seq, mut_seq, resno, func):
    devs = []
    for i in window_starts(resno, len(wt_seq)):
        wt_win = wt_seq[i:i + WINDOW]
        mut_win = mut_seq[i:i + WINDOW]
        devs.append(abs(func(mut_win) - func(wt_win)))
    return sum(devs) / len(devs)

#calculating BDE
_bde_lookup = None

def load_bde_table():
    global _bde_lookup
    if _bde_lookup is None:
        d = pd.read_csv(HERE / "dipeptide.csv", skipinitialspace=True, encoding="utf-8-sig")
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
    if resno - 2 >= 0:                       # left neighbour exists
        left = KELCH_SEQ[resno - 2]
        devs.append(abs(get_bde(left, mut) - get_bde(left, og)))
    if resno < len(KELCH_SEQ):               # right neighbour exists
        right = KELCH_SEQ[resno]
        devs.append(abs(get_bde(mut, right) - get_bde(og, right)))
    return sum(devs) / len(devs)


#compute all features from mean deviations
def compute_features(mutation):
    og, resno, mut = parse_mutation(mutation)
    mutant_seq = build_mutant(resno, mut)
    return {
        "II": mean_window_deviation(KELCH_SEQ, mutant_seq, resno, instability_index),
        "ALIN": mean_window_deviation(KELCH_SEQ, mutant_seq, resno, aliphatic_index),
        "BDE": bde_deviation(og, mut, resno),
    }


#loading config files
def load_config(name):
    path = HERE / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found. Run the training script to generate it, "
            f"and place it next to main.py.")
    with open(path) as f:
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

#known resistance lookup
def load_known():
    path = HERE / "resistance_lookup.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    known = {}
    for _, r in df.iterrows():
        name = str(r["Mutation"]).strip().upper()
        pc = r.get("PC")
        src = r.get("Source")
        known[name] = {
            "resistance": str(r["Resistance"]).strip(),
            "pc": None if pd.isna(pc) else float(pc),
            "source": None if pd.isna(src) else str(src).strip()
        }
    return known


def print_known(name, entry):
    print(f"\nMutation: {name}")
    print("Source  : reported data (resistance_lookup.csv); model not used")
    if entry["source"]:
        print(f"Reference: {entry['source']}")
    print(f"Resistance profile: {entry['resistance']}")
    if entry["pc"] is None:
        print("PC1/2 : not reported")
    else:
        print(f"PC1/2 : {entry['pc']}")


#run model
def run_model(name):
    try:
        feats = compute_features(name)
        clf_cfg = load_config("classifier_config.json")
        reg_cfg = load_config("regression_config.json")
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    label, prob = predict_resistance(clf_cfg, feats)
    half_life, interval = predict_half_life(reg_cfg, feats)

    print(f"Mutation: {name}")
    print("Source  : MODEL PREDICTION (not a reported experimental result)")
    print("Features (deviation from WT): " + ", ".join(f"{k}={v:.3f}" for k, v in feats.items()))

    print("\n--- Resistance classification ---")
    print(f"Prediction   : {label}")
    print(f"P(Resistant) : {prob:.3f}   (threshold = 0.4)")
    if abs(prob - clf_cfg["threshold"]) < LOW_MARGIN:
        print("Note: probability is close to the threshold; low-confidence call.")

    print("\n--- Clearance half-life (PC1/2) ---")
    print(f"Predicted PC1/2: {half_life:.2f}")
    if interval:
        print(f"95% prediction interval: {interval[0]:.2f} - {interval[1]:.2f}")
    print("Note: regression has weak predictive power (LOOCV Q2 = 0.11); treat as a rough estimate.")

    print("Research tool, not a clinical diagnostic.")


#main function for integration
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force_model = "--model" in sys.argv

    text = args[0] if args else input(
        "Enter mutation (format: <WT AA><residue number><mutant AA>, e.g. C580Y): ")
    name = text.strip().upper()

    try:
        parse_mutation(name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    entry = load_known().get(name)
    if entry:
        print_known(name, entry)
        if not force_model:
            return
        print("\n--- Model prediction, for comparison only ---")
    run_model(name)


if __name__ == "__main__":
    main()

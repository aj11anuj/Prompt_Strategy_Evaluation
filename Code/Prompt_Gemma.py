import random
import time
import re
from collections import Counter
from datasets import load_dataset
import ollama
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL = "gemma3:4b"
LANG = "swa"
N_SUBSAMPLES = 1 # Using the entire dataset for the test split (600 samples) makes increasing the subsample redundant.
SUBSAMPLE_SIZE = 600
SEEDS = [42, 123, 999]
OLLAMA_OPTIONS = {"temperature": 0, "num_predict": 64}
OUTPUT_CSV = "afrixnli_scriptaware_results.csv"

LABEL_MAP = {0: "entailment", 1: "contradiction", 2: "neutral"}
ALL_LABELS = ["entailment", "contradiction", "neutral"]
LANG_NAME = {"swa": "Swahili", "hau": "Hausa", "yor": "Yoruba"}.get(LANG, LANG)


def detect_script(text):
    arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    return "Arabic" if arabic_re.search(text) else "Latin"


def create_script_aware_prompt(premise, hypothesis, language_code):
    lang = LANG_NAME
    script = detect_script(premise + " " + hypothesis)

    if script == "Arabic":
        return f"""The following text is written in the Arabic-derived Ajami script used for {lang}. INSTRUCTIONS:
1) Internally transliterate the text into {lang} written in Latin script. Do NOT output it.
2) Reason in {lang}.
3) Decide the relationship.
4) Output exactly ONE English word: entailment, contradiction, or neutral.
Decision rules:
- entailment: premise makes hypothesis true.
- contradiction: premise makes hypothesis false.
- neutral: neither true nor false.
Premise (Ajami): "{premise}"
Hypothesis (Ajami): "{hypothesis}"
Answer:""".strip()
    else:
        return f"""You are a fluent {lang} speaker. INSTRUCTIONS:
1) Read and reason in {lang}.
2) Decide the relationship.
3) Output exactly ONE English word: entailment, contradiction, or neutral.
Decision rules:
- entailment: premise makes hypothesis true.
- contradiction: premise makes hypothesis false.
- neutral: neither true nor false.
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Answer:""".strip()


def extract_prediction_from_response(response_text):
    if not response_text:
        return None
    text = response_text.lower()
    for label in ALL_LABELS:
        if label in text:
            return label
    return None


def run_one_subsample(test_data, indices, model_name):
    records = []
    start = time.time()

    for idx in tqdm(indices, desc="samples", leave=False):
        item = test_data[idx]
        premise, hypothesis = item["premise"], item["hypothesis"]
        true_label = LABEL_MAP[int(item["label"])]
        script_type = detect_script(premise + " " + hypothesis)
        prompt = create_script_aware_prompt(premise, hypothesis, LANG)

        success, response_text, proc_time = False, None, None
        for _ in range(2):
            try:
                t0 = time.time()
                resp = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    options=OLLAMA_OPTIONS
                )
                proc_time = time.time() - t0
                response_text = resp.get("response", "").strip()
                success = True
                break
            except Exception:
                time.sleep(1)

        predicted = extract_prediction_from_response(response_text) if success else None

        records.append({
            "sample_idx": idx,
            "true_label": true_label,
            "predicted": predicted,
            "is_correct": predicted == true_label,
            "script": script_type,
            "success": success,
            "proc_time": proc_time,
            "raw_response": response_text
        })

    return records, time.time() - start


def compute_metrics(df):
    df_ok = df[df["success"]]
    if len(df_ok) == 0:
        return {}

    y_true = df_ok["true_label"]
    y_pred = df_ok["predicted"].fillna("neutral")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=ALL_LABELS)

    report = classification_report(
        y_true,
        y_pred,
        labels=ALL_LABELS,
        output_dict=True,
        zero_division=0
    )

    per_class_f1 = {lbl: report[lbl]["f1-score"] for lbl in ALL_LABELS}

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "n": len(df_ok),
        "avg_proc_time_s": df_ok["proc_time"].mean()
    }


def script_split_accuracy(df):
    out = {}
    for s in ["Arabic", "Latin"]:
        sub = df[(df["success"]) & (df["script"] == s)]
        out[s] = None if len(sub) == 0 else sub["is_correct"].mean()
    return out


def prediction_summary(df):
    counts = Counter(df["predicted"].dropna())
    for lbl in ALL_LABELS:
        counts.setdefault(lbl, 0)
    return dict(counts)


def main():
    print(f"\nLoading AfriXNLI ({LANG_NAME})")
    dataset = load_dataset("masakhane/afrixnli", name=LANG)
    test_data = dataset["test"]

    actual_sample_size = min(SUBSAMPLE_SIZE, len(test_data))

    all_dfs = []
    metrics_all = []
    ajami_accs, latin_accs = [], []

    per_class_f1_all = {lbl: [] for lbl in ALL_LABELS}

    for i in range(N_SUBSAMPLES):
        seed = SEEDS[i]
        rng = random.Random(seed)
        indices = rng.sample(range(len(test_data)), actual_sample_size)

        print(f"\n[Subsample {i+1}]")
        records, duration = run_one_subsample(test_data, indices, MODEL)
        df = pd.DataFrame(records)
        df["subsample"] = i + 1
        all_dfs.append(df)

        metrics = compute_metrics(df)
        metrics_all.append(metrics)

        for lbl in ALL_LABELS:
            per_class_f1_all[lbl].append(metrics["per_class_f1"][lbl])

        split_acc = script_split_accuracy(df)
        if split_acc["Arabic"] is not None:
            ajami_accs.append(split_acc["Arabic"])
        if split_acc["Latin"] is not None:
            latin_accs.append(split_acc["Latin"])

        print(f" Accuracy        : {metrics['accuracy']:.3f}")
        print(f" Macro-F1        : {metrics['macro_f1']:.3f}")
        print(" Per-class F1    :",
              f"Ent={metrics['per_class_f1']['entailment']:.3f},",
              f"Con={metrics['per_class_f1']['contradiction']:.3f},",
              f"Neu={metrics['per_class_f1']['neutral']:.3f}")
        print(f" Avg time/sample : {metrics['avg_proc_time_s']:.2f}s")
        print(f" Total time      : {duration:.1f}s")
        print(" Script accuracy : Ajami =", split_acc["Arabic"],
              "| Latin =", split_acc["Latin"])
        print(" Predictions     :", prediction_summary(df))
        print(" Gold labels     :", Counter(df["true_label"]))

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    accs = [m["accuracy"] for m in metrics_all]
    f1s = [m["macro_f1"] for m in metrics_all]
    times = [m["avg_proc_time_s"] for m in metrics_all]

    print("\n================ FINAL SUMMARY ================")
    print(f"Language : {LANG_NAME}")
    print(f"Model    : {MODEL}")
    print(f"Strategy : Script-aware prompting\n")

    print(f"Accuracy : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"Macro-F1 : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"Avg time : {np.mean(times):.2f} ± {np.std(times):.2f} s")

    print("Per-class F1:")
    for lbl in ALL_LABELS:
        vals = per_class_f1_all[lbl]
        print(f"  {lbl.capitalize():14}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    if ajami_accs:
        print(f"Ajami acc: {np.mean(ajami_accs):.3f} ± {np.std(ajami_accs):.3f}")
    if latin_accs:
        print(f"Latin acc: {np.mean(latin_accs):.3f} ± {np.std(latin_accs):.3f}")

    print("==============================================\n")


if __name__ == "__main__":
    main()

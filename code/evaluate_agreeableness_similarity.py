import argparse
import csv
import json

import numpy as np
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer

TARGET_WORDS = ("宜人性", "果干")


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_text(text, tokenizer, model, device):
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**encoded)
    pooled = mean_pooling(output.last_hidden_state, encoded["attention_mask"])
    return pooled.squeeze(0).cpu().numpy()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(np.mean(values)) if n else float("nan")
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    return {"mean": mean, "sd": sd, "n": n}


def group_values(rows, model_type, level, metric):
    return [row[metric] for row in rows if row["model_type"] == model_type and row["agreeableness_level"] == level]


def welch_t_test(group_a, group_b):
    result = stats.ttest_ind(group_a, group_b, equal_var=False)
    return {"t": float(result.statistic), "p": float(result.pvalue), "df": float(result.df)}


def print_comparison(label, rows, metric, group_a_key, group_b_key, label_a, label_b):
    values_a = group_values(rows, *group_a_key, metric)
    values_b = group_values(rows, *group_b_key, metric)
    if not values_a or not values_b:
        print(f"[{label}] skipped, empty group (n_a={len(values_a)}, n_b={len(values_b)})")
        return
    stat_a = summarize(values_a)
    stat_b = summarize(values_b)
    test = welch_t_test(values_a, values_b)
    print(f"[{label}] Mean({label_a}) = {stat_a['mean']:.3f}, SD = {stat_a['sd']:.3f} (n={stat_a['n']}); "
          f"Mean({label_b}) = {stat_b['mean']:.3f}, SD = {stat_b['sd']:.3f} (n={stat_b['n']}); "
          f"t({test['df']:.1f}) = {test['t']:.3f}, p = {test['p']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generation_comparison_results.json")
    parser.add_argument("--bert-model", default="bert-base-chinese")
    parser.add_argument("--output-csv", default="agreeableness_similarity_results.csv")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    # BERT-base mean pooling over a few dozen short texts is cheap; force CPU so this
    # evaluation never contends with a concurrently running training job for the single GPU.
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(device)
    model.eval()

    target_embeddings = {word: embed_text(word, tokenizer, model, device) for word in TARGET_WORDS}

    rows = []
    group_counters = {}
    for record in records:
        model_type = record["model_type"]
        level = record["agreeableness_level"]
        key = (model_type, level)
        sample_index = record.get("sample_index", group_counters.get(key, 0))
        group_counters[key] = sample_index + 1

        text_embedding = embed_text(record["generated_text"], tokenizer, model, device)
        rows.append({
            "model_type": model_type,
            "agreeableness_level": level,
            "sample_index": sample_index,
            "宜人性_similarity": cosine_similarity(text_embedding, target_embeddings["宜人性"]),
            "果干_similarity": cosine_similarity(text_embedding, target_embeddings["果干"]),
        })

    fieldnames = ["model_type", "agreeableness_level", "sample_index", "宜人性_similarity", "果干_similarity"]
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}\n")

    print("=== Group summary (mean / SD / n) ===")
    for metric in ("宜人性_similarity", "果干_similarity"):
        for model_type in ("custom", "non_custom"):
            for level in ("high", "low"):
                values = group_values(rows, model_type, level, metric)
                if not values:
                    continue
                stat = summarize(values)
                print(f"[{metric}] {model_type}/{level}: Mean = {stat['mean']:.3f}, SD = {stat['sd']:.3f}, n = {stat['n']}")

    print("\n=== t-tests: custom vs non_custom within high agreeableness_level (paper Table 4/5 comparison) ===")
    print_comparison("accuracy: 宜人性_similarity, high", rows, "宜人性_similarity",
                      ("custom", "high"), ("non_custom", "high"), "Custom LLM", "Non-custom LLM")
    print_comparison("contextual relevancy: 果干_similarity, high", rows, "果干_similarity",
                      ("custom", "high"), ("non_custom", "high"), "Custom LLM", "Non-custom LLM")

    print("\n=== t-test: custom model, high vs low agreeableness_level (manipulation check on 宜人性_similarity) ===")
    print_comparison("custom LLM: 宜人性_similarity, high vs low", rows, "宜人性_similarity",
                      ("custom", "high"), ("custom", "low"), "Custom LLM High", "Custom LLM Low")

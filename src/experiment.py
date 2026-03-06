"""
Experiment: Systematic Blindspots in LLM Self-Detection of Logical Fallacies

Three experiments:
1. Attribution Bias: Same fallacious text, framed as "self" vs "other" generated
2. Cross-Model Evaluation: Models evaluate own vs. other models' generated reasoning
3. Prompting Intervention: Explicit fallacy-checking instructions

Uses LOGIC dataset (13 fallacy types) and real OpenAI API calls.
"""

import os
import csv
import json
import random
import time
import asyncio
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Config
SAMPLES_PER_CATEGORY = 15  # examples per fallacy type
MODELS = ["gpt-4.1", "gpt-4o-mini"]
BASE_DIR = Path("/workspaces/llm-fallacy-detection-4626-claude")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def load_logic_dataset():
    """Load LOGIC dataset and sample examples per category."""
    data = []
    with open(BASE_DIR / "datasets/logical_fallacy/edu_train.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            label = row[-2]  # updated_label
            text = row[-3]   # source_article
            if text and label and len(text.strip()) > 10:
                data.append({"text": text.strip(), "label": label.strip()})

    # Group by category and sample
    by_category = defaultdict(list)
    for d in data:
        by_category[d["label"]].append(d)

    sampled = []
    for cat, items in sorted(by_category.items()):
        random.shuffle(items)
        sampled.extend(items[:SAMPLES_PER_CATEGORY])

    print(f"Loaded {len(sampled)} examples across {len(by_category)} categories")
    for cat in sorted(by_category):
        n = min(len(by_category[cat]), SAMPLES_PER_CATEGORY)
        print(f"  {cat}: {n} examples")
    return sampled


def call_llm(model, messages, temperature=0.0, max_tokens=300):
    """Call OpenAI API with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"API error after 3 attempts: {e}")
                return None


def parse_detection_response(response):
    """Parse LLM response to determine if fallacy was detected.
    Returns (detected: bool, fallacy_type: str or None)
    """
    if response is None:
        return None, None

    response_lower = response.lower()

    # Check for explicit "no fallacy" indicators
    no_fallacy_phrases = [
        "no logical fallacy", "no fallacy", "does not contain a fallacy",
        "does not contain a logical fallacy", "not a fallacy",
        "no logical error", "logically sound", "logically valid",
        "reasoning is valid", "reasoning is sound", "reasoning is correct",
    ]
    for phrase in no_fallacy_phrases:
        if phrase in response_lower:
            return False, None

    # Check for "yes fallacy" indicators
    yes_fallacy_phrases = [
        "contains a fallacy", "contains a logical fallacy", "logical fallacy",
        "is a fallacy", "this is an example of", "fallacy detected",
        "yes, this", "yes,", "yes.", "the fallacy", "commits the fallacy",
        "fallacious", "this reasoning contains",
    ]
    for phrase in yes_fallacy_phrases:
        if phrase in response_lower:
            return True, None

    # Default: if the response mentions any common fallacy name, consider detected
    fallacy_names = [
        "ad hominem", "straw man", "strawman", "appeal to", "false dilemma",
        "slippery slope", "circular reasoning", "red herring", "tu quoque",
        "false cause", "hasty generalization", "equivocation", "ad populum",
        "bandwagon", "begging the question",
    ]
    for name in fallacy_names:
        if name in response_lower:
            return True, name

    return False, None


# ============================================================
# EXPERIMENT 1: Attribution Bias
# ============================================================

def run_attribution_experiment(samples, model="gpt-4.1"):
    """Test if labeling reasoning as 'self' vs 'other' changes detection accuracy.
    Each sample is presented twice: once as self-generated, once as other-generated.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Attribution Bias ({model})")
    print(f"{'='*60}")

    results = []

    for i, sample in enumerate(samples):
        text = sample["text"]
        true_label = sample["label"]

        # Condition 1: Self-attributed
        self_prompt = [
            {"role": "system", "content": "You are an AI assistant evaluating reasoning for logical fallacies."},
            {"role": "user", "content": (
                f"Below is a piece of reasoning that you previously generated in an earlier conversation. "
                f"Please evaluate whether it contains a logical fallacy.\n\n"
                f"Your previous reasoning:\n\"{text}\"\n\n"
                f"Does this reasoning contain a logical fallacy? If yes, identify the type. "
                f"Start your answer with 'Yes' or 'No'."
            )}
        ]

        # Condition 2: Other-attributed
        other_prompt = [
            {"role": "system", "content": "You are an AI assistant evaluating reasoning for logical fallacies."},
            {"role": "user", "content": (
                f"Below is a piece of reasoning generated by a different AI model. "
                f"Please evaluate whether it contains a logical fallacy.\n\n"
                f"The other model's reasoning:\n\"{text}\"\n\n"
                f"Does this reasoning contain a logical fallacy? If yes, identify the type. "
                f"Start your answer with 'Yes' or 'No'."
            )}
        ]

        # Condition 3: Neutral (no attribution)
        neutral_prompt = [
            {"role": "system", "content": "You are an AI assistant evaluating reasoning for logical fallacies."},
            {"role": "user", "content": (
                f"Below is a piece of reasoning. "
                f"Please evaluate whether it contains a logical fallacy.\n\n"
                f"Reasoning:\n\"{text}\"\n\n"
                f"Does this reasoning contain a logical fallacy? If yes, identify the type. "
                f"Start your answer with 'Yes' or 'No'."
            )}
        ]

        self_resp = call_llm(model, self_prompt)
        other_resp = call_llm(model, other_prompt)
        neutral_resp = call_llm(model, neutral_prompt)

        self_detected, _ = parse_detection_response(self_resp)
        other_detected, _ = parse_detection_response(other_resp)
        neutral_detected, _ = parse_detection_response(neutral_resp)

        results.append({
            "text": text[:100],
            "true_label": true_label,
            "self_detected": self_detected,
            "other_detected": other_detected,
            "neutral_detected": neutral_detected,
            "self_response": self_resp[:200] if self_resp else None,
            "other_response": other_resp[:200] if other_resp else None,
            "neutral_response": neutral_resp[:200] if neutral_resp else None,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(samples)} samples...")

    # Save results
    with open(RESULTS_DIR / f"exp1_attribution_{model}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Completed {len(results)} samples")
    return results


# ============================================================
# EXPERIMENT 2: Cross-Model Evaluation
# ============================================================

def generate_reasoning(samples, model):
    """Have a model generate CoT reasoning about whether text is fallacious."""
    print(f"\n  Generating reasoning with {model}...")
    generated = []

    for i, sample in enumerate(samples):
        text = sample["text"]

        prompt = [
            {"role": "system", "content": "You are a logical reasoning assistant. Analyze arguments carefully."},
            {"role": "user", "content": (
                f"Analyze the following statement and explain your reasoning about "
                f"whether it is logically valid or contains errors:\n\n"
                f"\"{text}\"\n\n"
                f"Provide a step-by-step analysis of the reasoning."
            )}
        ]

        resp = call_llm(model, prompt, max_tokens=400)
        generated.append({
            "original_text": text,
            "true_label": sample["label"],
            "model": model,
            "reasoning": resp,
        })

        if (i + 1) % 20 == 0:
            print(f"    Generated {i+1}/{len(samples)}...")

    return generated


def cross_evaluate(generated_reasoning, evaluator_model):
    """Have evaluator_model assess whether generated reasoning contains fallacies."""
    print(f"  {evaluator_model} evaluating {len(generated_reasoning)} reasoning chains...")
    results = []

    for i, item in enumerate(generated_reasoning):
        reasoning = item["reasoning"]
        if reasoning is None:
            results.append({"detected": None, "response": None})
            continue

        prompt = [
            {"role": "system", "content": "You are an expert in logical reasoning and fallacy detection."},
            {"role": "user", "content": (
                f"The following reasoning was generated by an AI model analyzing a statement. "
                f"Does the ORIGINAL STATEMENT being analyzed contain a logical fallacy? "
                f"Focus on the original statement, not the analysis.\n\n"
                f"AI's analysis:\n{reasoning}\n\n"
                f"Based on this analysis, does the original statement contain a logical fallacy? "
                f"Answer 'Yes' or 'No' and briefly explain."
            )}
        ]

        resp = call_llm(evaluator_model, prompt)
        detected, _ = parse_detection_response(resp)

        results.append({
            "detected": detected,
            "response": resp[:200] if resp else None,
        })

        if (i + 1) % 20 == 0:
            print(f"    Evaluated {i+1}/{len(generated_reasoning)}...")

    return results


def run_crossmodel_experiment(samples):
    """Generate reasoning from both models, then cross-evaluate."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Cross-Model Evaluation")
    print(f"{'='*60}")

    # Use a subset for cross-model (more expensive)
    subset = samples[:100]  # 100 samples

    all_results = {}

    for gen_model in MODELS:
        generated = generate_reasoning(subset, gen_model)

        for eval_model in MODELS:
            key = f"{gen_model}_gen_{eval_model}_eval"
            eval_results = cross_evaluate(generated, eval_model)

            combined = []
            for gen, ev in zip(generated, eval_results):
                combined.append({
                    "original_text": gen["original_text"][:100],
                    "true_label": gen["true_label"],
                    "generator": gen_model,
                    "evaluator": eval_model,
                    "is_self_eval": gen_model == eval_model,
                    "reasoning": gen["reasoning"][:200] if gen["reasoning"] else None,
                    "detected": ev["detected"],
                    "eval_response": ev["response"],
                })

            all_results[key] = combined
            n_detected = sum(1 for c in combined if c["detected"] is True)
            n_valid = sum(1 for c in combined if c["detected"] is not None)
            rate = n_detected / n_valid if n_valid > 0 else 0
            print(f"  {key}: {n_detected}/{n_valid} detected ({rate:.1%})")

    with open(RESULTS_DIR / "exp2_crossmodel.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ============================================================
# EXPERIMENT 3: Prompting Intervention
# ============================================================

def run_intervention_experiment(samples, model="gpt-4.1"):
    """Test if explicit fallacy-checking prompt reduces self-attribution bias."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: Prompting Intervention ({model})")
    print(f"{'='*60}")

    # Use a subset
    subset = samples[:80]
    results = []

    for i, sample in enumerate(subset):
        text = sample["text"]
        true_label = sample["label"]

        # Self-attributed with intervention prompt
        intervention_prompt = [
            {"role": "system", "content": (
                "You are an expert logician tasked with finding logical fallacies. "
                "Be extra critical and thorough. Common fallacies include: ad hominem, "
                "straw man, appeal to emotion, false dilemma, circular reasoning, "
                "slippery slope, false causality, ad populum, equivocation, "
                "red herring, appeal to authority, hasty generalization, "
                "and tu quoque. Check carefully for EACH of these."
            )},
            {"role": "user", "content": (
                f"Below is reasoning you previously generated. Carefully check it "
                f"for ANY logical fallacies. Be thorough and critical.\n\n"
                f"Your previous reasoning:\n\"{text}\"\n\n"
                f"Does this contain a logical fallacy? Start with 'Yes' or 'No'."
            )}
        ]

        resp = call_llm(model, intervention_prompt)
        detected, _ = parse_detection_response(resp)

        results.append({
            "text": text[:100],
            "true_label": true_label,
            "intervention_detected": detected,
            "response": resp[:200] if resp else None,
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(subset)} samples...")

    with open(RESULTS_DIR / f"exp3_intervention_{model}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Completed {len(results)} samples")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("LLM Fallacy Self-Detection Experiment")
    print("=" * 60)
    print(f"Models: {MODELS}")
    print(f"Samples per category: {SAMPLES_PER_CATEGORY}")

    # Load data
    samples = load_logic_dataset()

    # Save config
    config = {
        "seed": SEED,
        "models": MODELS,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "total_samples": len(samples),
        "dataset": "LOGIC (Jin et al. 2022)",
        "categories": sorted(set(s["label"] for s in samples)),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments
    t0 = time.time()

    # Exp 1: Attribution bias
    exp1_results = run_attribution_experiment(samples, model="gpt-4.1")
    t1 = time.time()
    print(f"\nExp 1 took {t1-t0:.0f}s")

    # Exp 2: Cross-model
    exp2_results = run_crossmodel_experiment(samples)
    t2 = time.time()
    print(f"\nExp 2 took {t2-t1:.0f}s")

    # Exp 3: Intervention
    exp3_results = run_intervention_experiment(samples, model="gpt-4.1")
    t3 = time.time()
    print(f"\nExp 3 took {t3-t2:.0f}s")

    print(f"\nTotal experiment time: {t3-t0:.0f}s")
    print("Results saved to results/")


if __name__ == "__main__":
    main()

# Systematic Blindspots in LLM Self-Detection of Logical Fallacies: A Comparative Analysis

## 1. Executive Summary

We investigated whether large language models exhibit asymmetric detection rates for logical fallacies when evaluating reasoning attributed to themselves vs. other models. Using 195 fallacious reasoning examples from the LOGIC dataset (13 fallacy categories) and real API calls to GPT-4.1 and GPT-4o-mini, we conducted three experiments: (1) attribution framing, (2) cross-model evaluation, and (3) prompting intervention.

**Key finding**: Contrary to our hypothesis, GPT-4.1 does **not** show statistically significant self-evaluation bias. The attribution bias (other-attributed detection rate minus self-attributed) was only +3.1% (p = 0.146, McNemar's test), well below the hypothesized 20-40%. In genuine cross-model evaluation, self-evaluation was actually *slightly higher* than cross-evaluation (82.0% vs 79.5%). However, we found notable **category-specific asymmetries**: false dilemma, faulty generalization, and intentional fallacies showed +13.3% attribution bias, while fallacy of credibility showed -13.3% reverse bias.

**Practical implication**: Modern frontier LLMs (GPT-4.1) appear robust to simple attribution framing for fallacy detection, though category-specific blindspots exist that merit further investigation with larger samples.

## 2. Goal

**Hypothesis**: Large language models exhibit asymmetric detection rates for logical fallacies when evaluating their own chain-of-thought reasoning compared to other models' reasoning, with systematic blindspots correlating to fallacy types.

**Why important**: If models cannot reliably self-evaluate their reasoning, LLM-as-judge systems, self-correction pipelines, and safety mechanisms that rely on self-monitoring are fundamentally limited. Understanding whether self-attribution creates evaluation bias is critical for deploying trustworthy AI systems.

## 3. Data Construction

### Dataset Description
- **Source**: LOGIC dataset (Jin et al., 2022) — a foundational logical fallacy detection benchmark
- **Size**: 195 sampled examples (15 per category, 13 categories)
- **Categories**: ad hominem, ad populum, appeal to emotion, circular reasoning, equivocation, fallacy of credibility, fallacy of extension, fallacy of logic, fallacy of relevance, false causality, false dilemma, faulty generalization, intentional
- **All examples are confirmed fallacious reasoning** (positive class only)

### Example Samples

| Text | Fallacy Type |
|------|-------------|
| "The bigger a child's shoe size, the better the child's handwriting" | false causality |
| "Since many people believe this, then it must be true" | ad populum |
| "Senator Randall isn't lying when she says she cares about her constituents—she wouldn't lie to people she cares about." | circular reasoning |

### Data Quality
- All examples human-curated from educational sources
- No missing values in sampled set
- Balanced sampling: exactly 15 examples per category
- equivocation category has only 39 total examples in training set (smallest)

### Preprocessing
1. Loaded CSV, extracted `source_article` and `updated_label` columns
2. Filtered examples with text length > 10 characters
3. Stratified random sampling: 15 per category with seed=42

## 4. Experiment Description

### Methodology

#### High-Level Approach
Three experiments testing different aspects of self-evaluation bias:
1. **Attribution Bias**: Present identical fallacious text to GPT-4.1 under three framings (self-attributed, other-attributed, neutral)
2. **Cross-Model Evaluation**: Have GPT-4.1 and GPT-4o-mini generate CoT reasoning about fallacious texts, then cross-evaluate each other's outputs
3. **Prompting Intervention**: Test if explicit fallacy-checking instructions reduce self-attribution bias

#### Why This Method?
- Attribution framing (Exp 1) isolates the pure framing effect, controlling for content differences
- Cross-model evaluation (Exp 2) tests genuine self/other evaluation with naturally different reasoning styles
- Intervention (Exp 3) tests practical mitigation strategies
- This design was inspired by Tyen et al. (2024) who called for self-eval vs cross-eval studies, and Hong et al. (2024) who demonstrated category-specific blindspots

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| openai | latest | API access to GPT-4.1, GPT-4o-mini |
| numpy | 1.x | Numerical computation |
| pandas | 2.x | Data manipulation |
| scipy | 1.17.1 | Statistical tests |
| matplotlib | 3.x | Visualization |
| seaborn | 0.13.2 | Heatmaps |

#### Models
| Model | Role | API Calls |
|-------|------|-----------|
| GPT-4.1 | Primary evaluation model | ~955 |
| GPT-4o-mini | Secondary model for cross-evaluation | ~300 |

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| temperature | 0.0 | Deterministic for reproducibility |
| max_tokens | 300-400 | Sufficient for detection + explanation |
| seed | 42 | Reproducibility |
| samples_per_category | 15 | Balance between coverage and API cost |

### Experimental Protocol

#### Experiment 1: Attribution Bias (N=195)
Each fallacious text presented to GPT-4.1 in 3 conditions:
- **Self**: "Below is reasoning that you previously generated..."
- **Other**: "Below is reasoning generated by a different AI model..."
- **Neutral**: "Below is a piece of reasoning..."

All conditions ask: "Does this contain a logical fallacy? Start with Yes or No."

#### Experiment 2: Cross-Model Evaluation (N=100)
1. Each model generates CoT analysis of 100 fallacious texts
2. Each model evaluates both its own and the other model's analyses
3. Results form a 2x2 detection matrix

#### Experiment 3: Prompting Intervention (N=80)
Self-attributed condition with enhanced system prompt listing common fallacy types and instructing thorough checking.

#### Reproducibility
- Random seed: 42
- Temperature: 0.0 (deterministic)
- API models: gpt-4.1, gpt-4o-mini (OpenAI)
- Total API calls: ~1,255
- Total execution time: 3,141 seconds (~52 minutes)
- Hardware: CPU-only for API orchestration

### Raw Results

#### Experiment 1: Attribution Bias

| Condition | Detection Rate | N | 95% CI (Wilson) |
|-----------|---------------|---|-----------------|
| Self-attributed | 92.3% (180/195) | 195 | [87.7%, 95.3%] |
| Other-attributed | 95.4% (186/195) | 195 | [91.5%, 97.6%] |
| Neutral | 92.3% (180/195) | 195 | [87.7%, 95.3%] |

**Attribution bias**: +3.1% (Other - Self)
**McNemar's test**: p = 0.146 (not significant)
**Cohen's h**: 0.129 (small effect)

#### Experiment 1: Per-Category Results

| Category | Self | Other | Neutral | Bias (O-S) |
|----------|------|-------|---------|------------|
| ad hominem | 100.0% | 100.0% | 100.0% | 0.0% |
| ad populum | 93.3% | 100.0% | 100.0% | +6.7% |
| appeal to emotion | 80.0% | 80.0% | 73.3% | 0.0% |
| circular reasoning | 100.0% | 93.3% | 93.3% | -6.7% |
| equivocation | 93.3% | 100.0% | 93.3% | +6.7% |
| fallacy of credibility | 100.0% | 86.7% | 93.3% | **-13.3%** |
| fallacy of extension | 93.3% | 93.3% | 100.0% | 0.0% |
| fallacy of logic | 100.0% | 100.0% | 100.0% | 0.0% |
| fallacy of relevance | 86.7% | 93.3% | 93.3% | +6.7% |
| false causality | 93.3% | 93.3% | 93.3% | 0.0% |
| false dilemma | 86.7% | 100.0% | 86.7% | **+13.3%** |
| faulty generalization | 86.7% | 100.0% | 100.0% | **+13.3%** |
| intentional | 86.7% | 100.0% | 73.3% | **+13.3%** |

#### Experiment 2: Cross-Model Detection Matrix

| Generator \ Evaluator | GPT-4.1 | GPT-4o-mini |
|-----------------------|---------|-------------|
| GPT-4.1 | **83.0%** (self) | 79.0% (cross) |
| GPT-4o-mini | 80.0% (cross) | **81.0%** (self) |

- Self-eval average: 82.0%
- Cross-eval average: 79.5%
- Difference: -2.5% (self > cross, opposite to hypothesis)

#### Experiment 2: Per-Category (Self vs Cross)

| Category | Self | Cross | Bias |
|----------|------|-------|------|
| ad hominem | 93.3% | 90.0% | -3.3% |
| ad populum | 93.3% | 73.3% | **-20.0%** |
| appeal to emotion | 76.7% | 83.3% | +6.7% |
| circular reasoning | 76.7% | 76.7% | 0.0% |
| equivocation | 66.7% | 76.7% | +10.0% |
| fallacy of credibility | 93.3% | 90.0% | -3.3% |
| fallacy of extension | 70.0% | 60.0% | -10.0% |

#### Experiment 3: Prompting Intervention

| Condition | Detection Rate | N |
|-----------|---------------|---|
| Self (no intervention) | 93.8% | 80 |
| Self (with intervention) | 93.8% | 80 |

**Improvement**: 0.0% (p = 1.0, McNemar's test)

## 5. Result Analysis

### Key Findings

1. **No significant overall attribution bias**: GPT-4.1 detects fallacies at 92.3% when self-attributed and 95.4% when other-attributed (p = 0.146). The 3.1% difference is not statistically significant.

2. **Self-evaluation is NOT worse in cross-model setting**: In Experiment 2, models actually detected fallacies slightly *better* when evaluating their own reasoning (82.0%) compared to other models' reasoning (79.5%), contradicting the hypothesis.

3. **Category-specific asymmetries exist but are inconsistent**: Three categories (false dilemma, faulty generalization, intentional) show +13.3% attribution bias favoring other-attributed detection. But fallacy of credibility shows -13.3% reverse bias. No category exceeds the hypothesized 20% threshold in Exp 1.

4. **Prompting intervention has zero effect**: Explicit fallacy-checking instructions produced identical detection rates (93.8% vs 93.8%), suggesting GPT-4.1 already applies maximal effort regardless of prompt framing.

5. **Appeal to emotion is the hardest fallacy type**: Consistently lowest detection across conditions (73-80%), regardless of attribution.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|-----------|--------|----------|
| H1: Self-attribution reduces detection | **Not supported** | Bias = +3.1%, p = 0.146 |
| H2: Self-eval < cross-eval (genuine) | **Not supported** | Self = 82.0% > Cross = 79.5% |
| H3: Category-specific blindspots (>15%) | **Partially supported** | 3 categories at +13.3%, 0 at >15% |
| H4: Intervention reduces bias | **Not supported** | Zero improvement (p = 1.0) |

### Comparison to Prior Work

- **Tyen et al. (2024)** found GPT-4 at ~53% accuracy for mistake finding. Our detection rates (83-95%) are much higher, likely because our task is simpler (binary fallacy detection vs. step-level error location).
- **Hong et al. (2024)** found GPT-4 at 87.7% for fallacy identification. Our 92-95% is consistent given we use GPT-4.1 (newer model).
- **Payandeh et al. (2023)** found a self-detection paradox in GPT-3.5. We do NOT find this paradox in GPT-4.1, suggesting newer models may have mitigated this issue.
- **Richardson et al. (2025)** found more capable models produce more human-like fallacies. Our finding that GPT-4.1 shows no significant self-eval bias may indicate that capability improvements also reduce evaluation asymmetries.

### Surprises and Insights

1. **Self-attribution slightly improves detection for some categories**: For fallacy of credibility, self-attribution actually increased detection by 13.3%. This suggests models may apply extra scrutiny to "their own" output in some cases — a form of self-critique rather than self-confirmation.

2. **The neutral framing and self-framing produce identical overall rates** (92.3% each). Only the "other-attributed" framing shows a boost, suggesting a possible "other-evaluation boost" rather than a "self-evaluation penalty."

3. **Cross-model evaluation is harder regardless of attribution**: Detection rates in Exp 2 (79-83%) are lower than Exp 1 (92-95%). This is because Exp 2 involves evaluating an AI's *analysis* of a fallacy, which is more complex than evaluating the raw fallacious text directly.

4. **GPT-4.1 is remarkably robust to framing manipulations**: The model appears to evaluate reasoning quality based on content, not attribution.

### Error Analysis

**Most commonly missed fallacy types** (across all conditions):
- Appeal to emotion: 73-80% detection (20-27% miss rate)
- Intentional fallacies: 73-87% detection (varies heavily by condition)
- False dilemma: 87% self, 100% other (largest single-category gap)

**Why appeal to emotion is hardest**: These examples often involve legitimate emotional content that also serves a persuasive function, making the boundary between valid emotional appeal and fallacious reasoning ambiguous.

**Why intentional fallacies are inconsistent**: This category includes diverse examples (burden of proof shifts, unfalsifiable claims) that don't share strong surface features, making detection dependent on deeper reasoning.

### Limitations

1. **Sample size**: 15 examples per category limits statistical power for per-category analysis. A Category with +13.3% bias (2/15 difference) could easily arise by chance.

2. **Only two models**: We only tested GPT-4.1 and GPT-4o-mini, both from OpenAI. Cross-provider evaluation (e.g., Claude, Gemini) would strengthen generalizability.

3. **Simple attribution manipulation**: Our framing ("you previously generated" vs "a different AI model generated") may be too transparent. Models may not genuinely treat the text as self-generated.

4. **All positive examples**: We only tested fallacious reasoning (no non-fallacious controls). True detection accuracy requires both positive and negative examples.

5. **Temperature=0 reduces variance**: Deterministic outputs prevent measuring response variability. Multiple runs with temperature>0 would provide confidence intervals.

6. **Dataset bias**: LOGIC dataset contains educational examples, which may be "textbook-easy" compared to naturalistic fallacies in the wild.

7. **Exp 2 design confound**: In cross-model evaluation, difficulty varies because each model's analysis differs. Lower cross-eval rates may reflect the evaluator struggling with unfamiliar reasoning styles, not a self-evaluation advantage.

## 6. Conclusions

### Summary
GPT-4.1 does **not** exhibit statistically significant self-evaluation bias for logical fallacy detection. The hypothesized 20-40% reduction in self-attributed detection accuracy was not observed; the actual bias was only +3.1% (p = 0.146). Category-specific asymmetries exist (up to 13.3%) but are inconsistent in direction and not statistically robust. Modern frontier LLMs appear remarkably resistant to attribution framing manipulations in fallacy detection.

### Implications
- **For AI safety**: Simple self-evaluation mechanisms may be more reliable than feared for fallacy detection tasks, at least for frontier models
- **For LLM-as-judge systems**: Attribution framing does not significantly bias evaluation outcomes
- **For self-correction**: The bottleneck is not self-evaluation bias but absolute detection capability (GPT-4.1 still misses 5-20% of fallacies depending on type)
- **Category-specific weaknesses** (appeal to emotion, intentional fallacies) represent genuine blindspots worth addressing through training

### Confidence in Findings
- **High confidence**: No overall attribution bias (well-powered at N=195)
- **Moderate confidence**: Category-specific patterns (underpowered at N=15 per category)
- **Low confidence**: Generalizability to other models and naturalistic settings

## 7. Next Steps

### Immediate Follow-ups
1. **Scale per-category samples to 50+** for statistical power on category-specific blindspots
2. **Add non-fallacious controls** to compute precision alongside recall
3. **Test Claude, Gemini, and open-source models** (Llama-3, Mistral) for cross-provider comparison

### Alternative Approaches
- **Implicit self-evaluation**: Instead of telling the model the text is "yours," use actual conversation history where the model genuinely generated the reasoning
- **Adversarial fallacy injection**: Inject subtle fallacies into model-generated CoT and test self-detection
- **Confidence calibration**: Measure not just binary detection but confidence levels

### Open Questions
- Why does "other-attribution" slightly boost detection? Is this a social/politeness effect?
- Are older/weaker models (GPT-3.5, Llama-2) more susceptible to self-evaluation bias?
- Does the self-evaluation bias emerge for more subtle, harder-to-detect fallacies?
- Would genuine self-generated reasoning (not attributed) produce different results?

## References

1. Jin, Z., et al. (2022). "Logical Fallacy Detection." Findings of EMNLP. arXiv:2202.13758
2. Tyen, G., et al. (2024). "LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location." arXiv:2311.08516
3. Hong, R., et al. (2024). "A Closer Look at the Self-Verification Abilities of LLMs in Logical Reasoning." arXiv:2311.07954
4. Pan, L., et al. (2024). "Are LLMs Good Zero-Shot Fallacy Classifiers?" arXiv:2410.15050
5. Payandeh, A., et al. (2023). "How Susceptible Are LLMs to Logical Fallacies?" arXiv:2308.09853
6. Liu, Z., et al. (2024). "Self-Contradictory Reasoning Evaluation and Detection." arXiv:2311.09603
7. Richardson, K., et al. (2025). "Theory-Grounded Evaluation of Human-Like Fallacy Patterns in LLM Reasoning." arXiv:2506.11128
8. Helwe, C., et al. (2024). "MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection." NAACL. arXiv:2311.09761

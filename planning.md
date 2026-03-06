# Research Plan: Systematic Blindspots in LLM Self-Detection of Logical Fallacies

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly used for reasoning tasks, but their ability to critically evaluate the logical validity of their own outputs is poorly understood. If models exhibit systematic blindspots when self-evaluating—detecting fallacies less accurately in their own reasoning than in others'—this has serious implications for AI safety, self-correction systems, and the reliability of LLM-as-judge paradigms.

### Gap in Existing Work
Tyen et al. (2024) showed LLMs cannot find reasoning errors but can correct them given location. Hong et al. (2024) showed category-specific performance variation across 232 fallacy types. Payandeh et al. (2023) demonstrated a self-detection paradox where GPT-3.5 generates fallacies it cannot detect. However, **no study has systematically compared self-evaluation vs. cross-model evaluation accuracy** across fallacy categories, nor tested whether source attribution alone (labeling reasoning as "yours" vs. "another model's") biases detection.

### Our Novel Contribution
1. **Attribution bias experiment**: Test whether framing identical fallacious reasoning as "self-generated" vs. "other-generated" changes detection accuracy
2. **Cross-model evaluation matrix**: Have multiple real LLMs evaluate each other's reasoning, measuring per-category detection asymmetries
3. **Category-specific blindspot profiling**: Identify which fallacy types show the largest self-vs-other detection gaps

### Experiment Justification
- **Experiment 1 (Attribution Bias)**: Tests pure framing effect—does labeling reasoning as "yours" reduce detection accuracy?
- **Experiment 2 (Cross-Model Detection)**: Tests genuine cross-evaluation—do models detect others' fallacies better than their own?
- **Experiment 3 (Prompting Intervention)**: Tests whether explicit "check for fallacies" instructions reduce self-evaluation bias

## Research Question
Do LLMs exhibit asymmetric detection rates for logical fallacies when evaluating self-attributed vs. other-attributed reasoning, and do these blindspots correlate with fallacy category?

## Hypothesis Decomposition
H1: Models show lower fallacy detection accuracy when reasoning is attributed to "self" vs. "another model" (attribution bias)
H2: Models show lower detection accuracy on their own genuinely generated reasoning vs. other models' reasoning (genuine self-bias)
H3: Blindspot severity varies by fallacy category (some categories show >15% self-other gap)
H4: Explicit fallacy-checking prompts reduce but do not eliminate self-evaluation bias

## Proposed Methodology

### Approach
Use the LOGIC dataset (13 fallacy types, 1849 training examples) as the primary source of fallacious reasoning. Run three experiments:
1. Attribution framing experiment (same text, different source labels)
2. Cross-model genuine evaluation (models evaluate own vs. others' generated reasoning)
3. Prompting intervention ablation

### Models
- GPT-4.1 (via OpenAI API) — primary model
- GPT-4o-mini (via OpenAI API) — secondary model (represents different capability tier)

### Experimental Steps
1. Sample 20 examples per fallacy type from LOGIC dataset (13 types × 20 = 260 examples)
2. **Exp 1**: Present each example to GPT-4.1 in two conditions: self-attributed vs. other-attributed. Binary detection task.
3. **Exp 2**: Have each model generate reasoning about 100 topics, then cross-evaluate for fallacies
4. **Exp 3**: Repeat Exp 1 with explicit "check for logical fallacies" system prompt
5. Compute per-category detection accuracy, asymmetry metrics, and statistical tests

### Baselines
- Random baseline (50% binary detection)
- No-attribution baseline (neutral framing)
- Per-model overall detection accuracy

### Evaluation Metrics
- Per-category detection accuracy (precision, recall, F1)
- Attribution bias index: accuracy(other-attributed) - accuracy(self-attributed)
- Blindspot severity: categories where bias index > 15%
- McNemar's test for paired accuracy differences

### Statistical Analysis Plan
- McNemar's test for paired binary outcomes (self vs. other conditions)
- Bonferroni correction for 13 category comparisons
- Cohen's h for effect sizes on proportions
- 95% confidence intervals via Wilson score

## Expected Outcomes
- H1 supported: 10-30% lower detection when self-attributed
- H3 supported: certain categories (appeal to authority, false dilemma) show larger gaps
- Some categories may show no gap (ad hominem, likely well-represented in training)

## Timeline (~60 min total)
- Setup + data prep: 10 min
- Experiment 1 (attribution bias): 15 min
- Experiment 2 (cross-model): 15 min
- Experiment 3 (intervention): 10 min
- Analysis + visualization: 10 min

## Potential Challenges
- API rate limits → use batching, reasonable sample sizes
- Cost → ~500 API calls at ~$0.01 each = ~$5
- Model variance → run key conditions twice with different orderings

## Success Criteria
- Complete data collection for all 3 experiments
- Statistical tests with p-values for each hypothesis
- Per-category blindspot analysis with visualizations
- Clear conclusion on whether self-evaluation bias exists

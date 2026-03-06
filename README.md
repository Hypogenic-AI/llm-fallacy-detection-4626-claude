# LLM Self-Detection of Logical Fallacies

Do large language models exhibit blindspots when evaluating their own reasoning for logical fallacies? We tested this using 195 fallacious examples across 13 categories with GPT-4.1 and GPT-4o-mini.

## Key Findings

- **No significant self-evaluation bias**: GPT-4.1 detects fallacies at 92.3% when self-attributed vs. 95.4% when other-attributed (p=0.146, not significant)
- **Self-eval is not worse in cross-model setting**: Models detected fallacies slightly *better* on their own reasoning (82.0%) than others' (79.5%)
- **Category-specific asymmetries exist**: False dilemma, faulty generalization, and intentional fallacies show +13.3% other-attribution advantage
- **Prompting intervention has zero effect**: Explicit fallacy-checking instructions produced identical 93.8% detection rates
- **Appeal to emotion is hardest**: Consistently lowest detection (73-80%) across all conditions

## Project Structure

```
.
├── REPORT.md              # Full research report with methodology and analysis
├── planning.md            # Research plan and experimental design
├── src/
│   ├── experiment.py      # Main experiment script (3 experiments)
│   └── analysis.py        # Statistical analysis and visualization
├── results/
│   ├── config.json        # Experiment configuration
│   ├── exp1_attribution_gpt-4.1.json    # Exp 1 raw results
│   ├── exp2_crossmodel.json              # Exp 2 raw results
│   ├── exp3_intervention_gpt-4.1.json   # Exp 3 raw results
│   ├── analysis_summary.json             # Combined analysis
│   └── plots/
│       ├── exp1_attribution_bias.png
│       ├── exp2_crossmodel.png
│       ├── exp3_intervention.png
│       └── summary.png
├── datasets/              # Pre-downloaded datasets (LOGIC, FALLACIES, etc.)
├── papers/                # Reference papers
└── literature_review.md   # Literature review
```

## Reproducing Results

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv pip install openai numpy pandas matplotlib scipy scikit-learn seaborn

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiments (~52 min, ~1255 API calls)
python src/experiment.py

# Run analysis
python src/analysis.py
```

## See Also

- [Full Report](REPORT.md) for detailed methodology, results, and analysis
- [Planning Document](planning.md) for experimental design rationale

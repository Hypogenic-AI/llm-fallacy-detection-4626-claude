# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "Systematic Blindspots in LLM Self-Detection of Logical Fallacies: A Comparative Analysis of Chain-of-Thought Reasoning Evaluation."

## Papers
Total papers downloaded: 19

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Are LLMs Good Zero-Shot Fallacy Classifiers? | Pan et al. | 2024 | papers/pan2024_llms_zero_shot_fallacy_classifiers.pdf | Zero-shot fallacy classification with CoT |
| 2 | When LLMs Meet Cunning Texts | Li et al. | 2024 | papers/li2024_when_llms_meet_cunning_texts.pdf | Fallacy understanding benchmark |
| 3 | CoCoLoFa Dataset | Yeh et al. | 2024 | papers/yeh2024_cocolofa_dataset.pdf | 7,706 comments, 8 fallacy types |
| 4 | Logical Fallacy Detection | Jin et al. | 2022 | papers/jin2022_logical_fallacy_detection.pdf | LOGIC dataset, foundational |
| 5 | MAFALDA Benchmark | Helwe et al. | 2023 | papers/helwe2023_mafalda_benchmark.pdf | Unified fallacy taxonomy |
| 6 | How Susceptible Are LLMs to Fallacies? | Sourati et al. | 2023 | papers/sourati2023_how_susceptible_llms_fallacies.pdf | LLM susceptibility testing |
| 7 | Reason from Fallacy | Ye et al. | 2024 | papers/ye2024_reason_from_fallacy.pdf | Fallacy understanding for reasoning |
| 8 | Boosting Fallacy Reasoning via Structure Tree | Jiang et al. | 2024 | papers/jiang2024_boosting_fallacy_reasoning_structure_tree.pdf | Structure-based approach |
| 9 | Case-Based Reasoning for Fallacies | Holtermann et al. | 2023 | papers/holtermann2023_case_based_reasoning_fallacies.pdf | CBR for fallacy classification |
| 10 | LLMs Cannot Find Reasoning Errors | Tyen et al. | 2024 | papers/valmeekam2023_llms_cannot_find_reasoning_errors.pdf | **KEY**: Mistake finding vs correction |
| 11 | Self-Verification in Logical Reasoning | Hong et al. | 2024 | papers/hong2023_self_verification_logical_reasoning.pdf | **KEY**: 232-type fallacy taxonomy |
| 12 | Self-Contradictory Reasoning | Liu et al. | 2023 | papers/liu2023_self_contradictory_reasoning.pdf | Self-contradiction detection |
| 13 | Faithful CoT Reasoning | Lyu et al. | 2023 | papers/lyu2023_faithful_cot_reasoning.pdf | CoT faithfulness |
| 14 | Measuring CoT Faithfulness | Arcuschin et al. | 2025 | papers/arcuschin2025_measuring_cot_faithfulness.pdf | Unlearning-based faithfulness |
| 15 | Critical-Questions-of-Thought | Karanikolas et al. | 2024 | papers/karanikolas2024_critical_questions_of_thought.pdf | Argumentative querying |
| 16 | LogicAsker | Xiao et al. | 2024 | papers/xiao2024_logicasker.pdf | Logical reasoning evaluation |
| 17 | RuozhiBench | Zhai et al. | 2025 | papers/zhai2025_ruozhibench.pdf | Misleading premises benchmark |
| 18 | Theory-Grounded Fallacy Patterns | Richardson et al. | 2025 | papers/richardson2025_theory_grounded_fallacy_patterns.pdf | Cognitive science grounding |
| 19 | Evaluation of LLM in Logical Fallacies | Feyisetan et al. | 2024 | papers/feyisetan2024_evaluation_llm_logical_fallacies.pdf | Practical LLM evaluation |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 8

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| LOGIC | HuggingFace/GitHub | 3,761 examples, 13 types | Classification | datasets/logical_fallacy/ | Foundational benchmark |
| FALLACIES | GitHub | 4,640 steps, 232 types | Binary ID + Classification | datasets/fallacies_232/ | Hierarchical taxonomy |
| MAFALDA | GitHub | 200 texts, 25 types | Span-level detection | datasets/mafalda/ | Unified benchmark |
| CoCoLoFa | GitHub | 7,706 comments, 8 types | Detection + Classification | datasets/cocolofa/ | LLM-assisted crowd-sourced |
| BIG-Bench Mistake | GitHub | 2,186 CoT traces | Error location | datasets/bigbench_mistake/ | Step-level annotation |
| RuozhiBench | GitHub | 675 questions, 16 types | Fallacy detection | datasets/ruozhibench/ | Bilingual |
| MMLU Logical Fallacies | HuggingFace | 162 MCQ | Multiple choice | datasets/mmlu_logical_fallacies/ | Standard benchmark |
| LogicClimate | GitHub | Climate claims | Classification | datasets/logic_climate/ | Domain-specific |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 7

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| MAFALDA | github.com/ChadiHelwe/MAFALDA | Unified benchmark + eval | code/MAFALDA/ | Contains eval scripts |
| logical-fallacy | github.com/causalNLP/logical-fallacy | LOGIC dataset + models | code/logical-fallacy/ | Foundational code |
| cocolofa | github.com/Crowd-AI-Lab/cocolofa | CoCoLoFa dataset | code/cocolofa/ | Data files in repo |
| logical-fallacy-identification | github.com/usc-isi-i2/logical-fallacy-identification | Multi-method baselines | code/logical-fallacy-identification/ | CBR, Prototex, etc. |
| FALLACIES | github.com/Raising-hrx/FALLACIES | 232-type fallacy dataset | code/FALLACIES/ | Key for our experiments |
| BIG-Bench-Mistake | github.com/WHGTyen/BIG-Bench-Mistake | CoT error traces | code/BIG-Bench-Mistake/ | Step-level annotations |
| ruozhibench | github.com/LibrAIResearch/ruozhibench | Misleading premises eval | code/ruozhibench/ | Bilingual + eval code |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for two queries covering fallacy detection and self-evaluation
2. Supplemented with arXiv API searches for specific paper titles
3. Searched HuggingFace, GitHub, and web for datasets and code repositories
4. Downloaded and verified all resources

### Selection Criteria
- Papers directly studying LLM fallacy detection or self-evaluation of reasoning
- Datasets with fallacy type annotations enabling category-specific analysis
- Code repositories with evaluation frameworks reusable for our experiments
- Preference for recent work (2023-2025) with established benchmarks

### Challenges Encountered
- Some HuggingFace dataset names required searching (MAFALDA and CoCoLoFa not directly on HF)
- Paper-finder timed out on one search; used fallback results
- No existing paper directly studies self-eval vs cross-eval for fallacy detection (confirming the research gap)

### Gaps and Workarounds
- No existing dataset specifically designed for self vs cross-model evaluation; will need to generate CoT reasoning from multiple models and create evaluation pairs
- Limited datasets for some fallacy types (e.g., equivocation only 58 examples in LOGIC)
- No standardized metric for "blindspot" measurement; will need to define detection asymmetry metric

## Recommendations for Experiment Design

### Primary Datasets
1. **FALLACIES (Hong et al.)**: Best for category-specific blindspot analysis (232 types, hierarchical taxonomy, both correct and fallacious steps)
2. **LOGIC (Jin et al.)**: Standard 13-type benchmark for overall detection performance
3. **BIG-Bench Mistake**: For studying error detection in actual CoT reasoning traces

### Experiment Architecture
1. **Phase 1 - Fallacy Instance Generation**: Use multiple LLMs (GPT-4, Claude, Llama, etc.) to generate CoT reasoning that may contain fallacies from FALLACIES dataset prompts
2. **Phase 2 - Self-Evaluation**: Have each model evaluate its own CoT reasoning for fallacies
3. **Phase 3 - Cross-Evaluation**: Have each model evaluate other models' CoT reasoning
4. **Phase 4 - Asymmetry Analysis**: Compare self-eval vs cross-eval accuracy per fallacy type

### Baseline Methods
1. Zero-shot binary detection (correct/incorrect)
2. Zero-shot with CoT prompting for detection
3. Per-fallacy-type accuracy breakdown
4. Random baseline

### Evaluation Metrics
1. **Detection accuracy** per fallacy type (self vs cross)
2. **Detection asymmetry** = cross-model accuracy - self-evaluation accuracy
3. **Blindspot score** = fallacy types where self-eval accuracy < cross-eval accuracy by >20%
4. **Category-level analysis** (formal vs informal subcategories)

### Code to Adapt/Reuse
1. **FALLACIES repo**: Taxonomy structure, evaluation framework
2. **MAFALDA repo**: Evaluation pipeline for multiple models
3. **BIG-Bench-Mistake repo**: Step-level mistake finding methodology
4. **logical-fallacy-identification repo**: Baseline classification methods

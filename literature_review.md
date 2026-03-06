# Literature Review: Systematic Blindspots in LLM Self-Detection of Logical Fallacies

## Research Area Overview

This review covers the intersection of two active research areas: (1) **logical fallacy detection by LLMs**, and (2) **LLM self-verification and reasoning error detection**. The central question is whether LLMs exhibit asymmetric detection rates when evaluating their own reasoning vs. other models' reasoning, with systematic blindspots correlating to fallacy types.

## Key Papers

### 1. LLMs Cannot Find Reasoning Errors, but Can Correct Them Given the Error Location
- **Authors**: Tyen, Mansoor, Carbune, Chen, Mak
- **Year**: 2024 (arXiv: 2311.08516)
- **Key Contribution**: Decomposes self-correction into *mistake finding* and *output correction*, showing LLMs fail at the former but succeed at the latter.
- **Methodology**: Created BIG-Bench Mistake dataset (2186 CoT traces from PaLM 2 Unicorn) across 5 tasks. Tested GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, Gemini Pro, PaLM 2 with 3 prompting methods (direct trace-level, direct step-level, CoT step-level).
- **Key Results**:
  - Best model (GPT-4) achieved only **52.87% accuracy** on mistake finding with direct step-level prompting
  - GPT-3.5-Turbo: ~14% accuracy overall
  - Authors note: "cross-model evaluation vs self-evaluation" difference needs further study (PaLM 2 evaluating its own traces may produce biased results)
  - Backtracking with oracle mistake location improves correctness significantly
- **Dataset**: BIG-Bench Mistake (5 tasks, 2186 traces) - **Available at**: github.com/WHGTyen/BIG-Bench-Mistake
- **Relevance**: **CRITICAL** - Directly demonstrates LLMs' inability to find reasoning errors, explicitly calls for studying self-eval vs cross-eval differences.

### 2. A Closer Look at the Self-Verification Abilities of LLMs in Logical Reasoning
- **Authors**: Hong, Zhang, Pang, Yu, Zhang
- **Year**: 2024 (arXiv: 2311.07954)
- **Key Contribution**: FALLACIES dataset with 232 fallacy types in hierarchical taxonomy (Formal: proposition, quantification, syllogism, probability; Informal: ambiguity, inconsistency, irrelevance, insufficiency, inappropriate presumption). Tests verification abilities across many LLMs.
- **Key Results**:
  - Most LLMs achieve 60-80% accuracy on binary fallacy identification; GPT-4 best at **87.7%**
  - **Formal fallacies much harder than informal** for most models (GPT-3.5: 74.1% formal vs 87.9% informal)
  - **Remarkably imbalanced performance across fallacy types** (e.g., Qwen-14B: 91.7% on inconsistency but 67.5% on probability)
  - Fallacy classification (232 types): GPT-4 only **35.0%** accuracy; most models <10%
  - Providing fallacy definitions **hurts** performance for most models (Vicuna-13B drops 13.5%)
- **Dataset**: FALLACIES (4,640 steps, 232 fallacy types) - **Available at**: github.com/Raising-hrx/FALLACIES
- **Relevance**: **CRITICAL** - Demonstrates category-specific blindspots, provides taxonomy and dataset for our experiments.

### 3. Are LLMs Good Zero-Shot Fallacy Classifiers?
- **Authors**: Pan, Wu, Li, Luu
- **Year**: 2024 (arXiv: 2410.15050)
- **Key Contribution**: Systematic evaluation of LLMs for zero-shot fallacy classification using diverse prompting schemes (single-round and multi-round with extraction, summarization, CoT).
- **Methodology**: Tested on benchmark datasets (LOGIC, LogicClimate, Argotario, others) with multi-round prompting including extraction, summarization, and CoT reasoning.
- **Key Results**: LLMs achieve acceptable zero-shot performance vs full-shot baselines; outperform in OOD scenarios; multi-round prompting improves performance.
- **Datasets Used**: LOGIC (13 classes), LogicClimate, Argotario (5 classes), others
- **Relevance**: Provides baselines for zero-shot fallacy classification and demonstrates CoT-based approaches.

### 4. MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification
- **Authors**: Helwe, Calamai, Paris, Clavel, Suchanek
- **Year**: 2024 (NAACL, arXiv: 2311.09761)
- **Key Contribution**: Unified fallacy benchmark consolidating previous datasets; refined taxonomy; evaluates LLMs under zero-shot setting.
- **Dataset**: 200 texts with span-level annotations, 25 fallacy types - **Available at**: github.com/ChadiHelwe/MAFALDA
- **Relevance**: Provides unified taxonomy and multi-source benchmark for fallacy evaluation.

### 5. Logical Fallacy Detection (Jin et al., 2022)
- **Authors**: Jin, Lalwani, Vaidhya, Shen, et al.
- **Year**: 2022 (Findings of EMNLP, arXiv: 2202.13758)
- **Key Contribution**: Proposed the task of logical fallacy detection; created LOGIC dataset (13 fallacy types from educational sources) and LogicClimate (climate change claims).
- **Key Results**: Pretrained LLMs perform poorly; structure-aware classifier outperforms best LM by 5.46% F1 on LOGIC.
- **Dataset**: LOGIC (2,680 train/511 test/570 dev, 13 categories) - **Available at**: github.com/causalNLP/logical-fallacy and HuggingFace (tasksource/logical-fallacy)
- **Relevance**: Foundational dataset and task definition for fallacy detection.

### 6. How Susceptible Are LLMs to Logical Fallacies?
- **Authors**: Payandeh, Sourati, Venkatesh, Deshpande, Rawat, Ilyas, Zhu, Kalyan
- **Year**: 2023 (arXiv: 2308.09853)
- **Key Contribution**: LOGICOM multi-agent debate benchmark testing LLM susceptibility to fallacious arguments. GPT-4 is **69% more susceptible** to fallacious vs logical arguments (worse than GPT-3.5's 41%). GPT-3.5 generates fallacies it cannot detect as a debater - a **self-detection paradox**.
- **Key Results**: Models are more easily swayed by fallacious arguments than logical ones; larger models not necessarily more robust
- **Relevance**: **CRITICAL** - Directly demonstrates self-detection paradox: models produce fallacies they cannot identify, supporting our asymmetric detection hypothesis.

### 7. Self-Contradictory Reasoning Evaluation and Detection
- **Authors**: Liu, Lee, Du, Sanyal, Zhao
- **Year**: 2024 (arXiv: 2311.09603)
- **Key Contribution**: Documents self-contradictory reasoning rates up to 50% (Mistral 7B on WinoBias). GPT-4 detects self-contradictions at only 52.2% F1 vs 66.7% for humans. "Begging the question" is the dominant fallacy pattern and is resistant to few-shot mitigation.
- **Key Results**: Self-contradiction rates vary significantly by model and task; even GPT-4 struggles to detect its own contradictions
- **Relevance**: **CRITICAL** - Directly demonstrates LLMs' poor self-detection of reasoning errors, with quantified human-LLM gap.

### 8. Faithful Chain-of-Thought Reasoning (Lyu et al., 2023)
- **Authors**: Lyu et al.
- **Year**: 2023 (arXiv: 2301.13379)
- **Key Contribution**: Proposes Faithful CoT, translating reasoning into symbolic language executed by deterministic solvers. Outperforms standard CoT on 8/10 benchmarks (gains up to +21.4%), establishing that **standard CoT is unfaithful to actual reasoning**.
- **Relevance**: Demonstrates CoT can contain hidden errors not reflected in outputs - relevant to understanding when self-evaluation may miss fallacies.

### 9. Measuring Chain of Thought Faithfulness by Unlearning Reasoning Steps
- **Authors**: Arcuschin et al.
- **Year**: 2025 (arXiv: 2502.14829)
- **Key Contribution**: Introduces FUR metric measuring CoT faithfulness via unlearning. Finds CoTs are more faithful than contextual methods suggest, but **faithfulness does not correlate with plausibility** (r=0.15). Pearson correlation between unlearning efficacy and prediction change is 0.889.
- **Relevance**: Provides methodology for measuring whether stated reasoning actually drives outputs; low faithfulness-plausibility correlation suggests models may miss errors in plausible-sounding reasoning.

### 10. Theory-Grounded Evaluation of Human-Like Fallacy Patterns in LLM Reasoning
- **Authors**: Richardson, Kearns, Moss, Wang-Mascianica, Koralus
- **Year**: 2025 (arXiv: 2506.11128)
- **Key Contribution**: Tests 38 models on 383 ETR-generated reasoning problems. More capable models produce **more human-like fallacies** (rho=0.360, p=0.0265) while overall correctness shows **NO correlation** with capability (r=0.004). Premise order reversal blocks 20-88% of fallacies.
- **Key Results**: Scaling model capability does not eliminate fallacy blindspots - it makes errors more systematically human-like
- **Relevance**: **CRITICAL** - Shows blindspots are systematic and correlated with model capability, not random; supports hypothesis that training characteristics shape fallacy patterns.

### 11. CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies
- **Authors**: Yeh, Wan, Huang
- **Year**: 2024 (EMNLP, arXiv: 2410.03457)
- **Key Contribution**: LLM-assisted crowd-sourced dataset of 7,706 news comments with 8 fallacy types. BERT fine-tuned achieves F1=0.86 detection, outperforming LLMs.
- **Dataset**: 5,370 train / test comments across 8 fallacy types - **Available at**: github.com/Crowd-AI-Lab/cocolofa
- **Relevance**: Provides ecologically valid fallacy examples and LLM performance baselines.

### 12. RuozhiBench: Evaluating LLMs with Logical Fallacies and Misleading Premises
- **Authors**: Zhai et al.
- **Year**: 2025 (arXiv: 2502.13125)
- **Key Contribution**: Bilingual benchmark (675 questions) with deceptive reasoning. Best model (Claude-3-haiku) achieved only 62% vs human upper bound >90%.
- **Dataset**: 675 questions, 16 categories - **Available at**: github.com/LibrAIResearch/ruozhibench
- **Relevance**: Tests LLMs' ability to identify misleading reasoning - directly relevant to blindspot detection.

## Common Methodologies

1. **Zero-shot prompting** for fallacy detection/classification: Used across most papers (Pan et al., Hong et al., Jin et al.)
2. **Multi-round prompting with CoT**: Pan et al. show extraction -> summarization -> classification chains improve detection
3. **Step-level verification**: Tyen et al. and Hong et al. evaluate reasoning step-by-step rather than holistically
4. **Contrastive evaluation**: Hong et al. use fallacious vs. corrected step pairs
5. **Backtracking correction**: Tyen et al. show correction given oracle error location works well

## Standard Baselines

- **Random baseline**: 50% for binary detection, 0.4% for 232-type classification
- **GPT-4**: Best performing across most benchmarks (87.7% identification, 35% classification)
- **GPT-3.5**: Strong baseline (81% identification)
- **Fine-tuned BERT**: Strong for in-domain detection (F1=0.86 on CoCoLoFa)
- **Structure-aware classifiers**: Jin et al.'s classifier outperforms LLMs on LOGIC dataset

## Evaluation Metrics

- **Accuracy**: Per-fallacy-type and overall (macro-averaged)
- **F1 Score**: Weighted average for detection tasks
- **Mistake location accuracy**: Exact match of error step (Tyen et al.)
- **Per-category breakdown**: Essential for identifying blindspots

## Datasets in the Literature

| Dataset | Types | Size | Source | Used By |
|---------|-------|------|--------|---------|
| LOGIC | 13 | 3,761 | Jin et al. 2022 | Pan, Jin, multiple |
| FALLACIES | 232 | 4,640 | Hong et al. 2024 | Hong et al. |
| MAFALDA | 25 | 200 texts | Helwe et al. 2024 | Helwe et al. |
| CoCoLoFa | 8 | 7,706 | Yeh et al. 2024 | Yeh et al. |
| BIG-Bench Mistake | 5 tasks | 2,186 | Tyen et al. 2024 | Tyen et al. |
| RuozhiBench | 16 | 675 | Zhai et al. 2025 | Zhai et al. |
| MMLU Logical Fallacies | ~20 | 162 | MMLU subset | Multiple |
| Argotario | 5 | ~2K | Habernal et al. 2018 | Pan, others |

## Gaps and Opportunities

1. **Self-eval vs cross-eval comparison**: Tyen et al. explicitly call for this but no one has done it systematically. Our hypothesis directly addresses this gap.
2. **Category-specific blindspot analysis**: Hong et al. show imbalanced performance but don't study whether blindspots correlate with training data characteristics.
3. **CoT-embedded fallacies**: Most work tests detection of standalone fallacious statements, not fallacies embedded within a model's own chain-of-thought reasoning.
4. **Asymmetry in self vs other evaluation**: No systematic study comparing detection rates when a model evaluates its own reasoning vs another model's reasoning on identical fallacy instances.

## Recommendations for Our Experiment

### Recommended Datasets
1. **FALLACIES (Hong et al.)**: 232 types with hierarchical taxonomy - best for studying category-specific blindspots
2. **LOGIC (Jin et al.)**: 13-type classification benchmark - widely used baseline
3. **BIG-Bench Mistake (Tyen et al.)**: CoT traces with annotated errors - for step-level verification experiments
4. **CoCoLoFa**: Ecologically valid fallacy examples in naturalistic text

### Recommended Baselines
1. Zero-shot GPT-4/GPT-3.5/Claude/Llama detection accuracy
2. Self-evaluation vs cross-model evaluation comparison
3. Per-fallacy-type accuracy breakdown
4. Random baseline (50% binary, category-dependent for classification)

### Recommended Metrics
1. **Per-fallacy-type accuracy** (macro-averaged across types)
2. **Self-detection rate vs cross-detection rate** (the key asymmetry metric)
3. **Category-level accuracy** (formal vs informal, and subcategories)
4. **Detection accuracy gap** = cross-model accuracy - self-evaluation accuracy (per fallacy type)

### Methodological Considerations
- Use **identical fallacy instances** for self-eval and cross-eval to control for difficulty
- Generate CoT reasoning from multiple models, then have each model evaluate both its own and others' reasoning
- Include Hong et al.'s fallacy taxonomy for systematic blindspot analysis
- Test with and without CoT prompting for detection
- Measure both binary detection (correct/incorrect) and classification (fallacy type)

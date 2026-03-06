# Downloaded Datasets

This directory contains datasets for the LLM fallacy detection research project. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: LOGIC (Jin et al., 2022)

### Overview
- **Source**: HuggingFace (tasksource/logical-fallacy) / GitHub (causalNLP/logical-fallacy)
- **Size**: 3,761 examples (2,680 train / 511 test / 570 dev)
- **Format**: HuggingFace Dataset / CSV
- **Task**: Multi-class fallacy classification (13 types)
- **License**: Research use
- **Fallacy Types**: ad hominem, ad populum, appeal to emotion, circular reasoning, equivocation, fallacy of credibility, fallacy of extension, fallacy of logic, fallacy of relevance, false causality, false dilemma, faulty generalization, intentional

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset
dataset = load_dataset("tasksource/logical-fallacy")
dataset.save_to_disk("datasets/logical_fallacy")
```

**From CSV (already in repo):**
```python
import pandas as pd
train = pd.read_csv("datasets/logical_fallacy/edu_train.csv")
```

### Label Distribution (train)
- faulty generalization: 401
- intentional: 321
- ad hominem: 289
- appeal to emotion: 217
- false causality: 212
- ad populum: 209
- fallacy of credibility: 200
- fallacy of logic: 176
- fallacy of relevance: 175
- false dilemma: 143
- circular reasoning: 140
- fallacy of extension: 139
- equivocation: 58

---

## Dataset 2: FALLACIES (Hong et al., 2024)

### Overview
- **Source**: GitHub (Raising-hrx/FALLACIES)
- **Size**: 4,640 reasoning steps covering 232 fallacy types
- **Format**: JSONL
- **Task**: Binary identification (correct/fallacious) and multi-class classification (232 types)
- **Taxonomy**: Hierarchical - Formal (proposition, quantification, syllogism, probability) / Informal (ambiguity, inconsistency, irrelevance, insufficiency, inappropriate presumption)

### Download Instructions
```bash
git clone https://github.com/Raising-hrx/FALLACIES.git
cp FALLACIES/step_fallacy.test.jsonl datasets/fallacies_232/
cp FALLACIES/fallacy_taxonomy.json datasets/fallacies_232/
```

### Loading
```python
import json
with open("datasets/fallacies_232/step_fallacy.test.jsonl") as f:
    data = [json.loads(line) for line in f]
# Each entry: {"step": "...", "entity": "...", "fallacy": "...", "label": 0/1}
```

---

## Dataset 3: MAFALDA (Helwe et al., 2024)

### Overview
- **Source**: GitHub (ChadiHelwe/MAFALDA)
- **Size**: 200 texts with 272 fallacy span annotations, 25 fallacy types
- **Format**: JSONL (span-level annotations: [start, end, label])
- **Task**: Span-level fallacy detection and classification

### Download Instructions
```bash
git clone https://github.com/ChadiHelwe/MAFALDA.git
cp MAFALDA/datasets/gold_standard_dataset.jsonl datasets/mafalda/
```

### Key Fallacy Types
hasty generalization (28), causal oversimplification (23), Appeal to Ridicule (20), false dilemma (18), ad hominem (16), ad populum (14), false causality (13), straw man (13), and others.

---

## Dataset 4: CoCoLoFa (Yeh et al., 2024)

### Overview
- **Source**: GitHub (Crowd-AI-Lab/cocolofa)
- **Size**: 7,706 comments (5,370 train / test / dev) across 452+ news articles
- **Format**: JSON
- **Task**: Fallacy detection and classification (8 types + none)
- **Fallacy Types**: slippery slope, appeal to worse problems, appeal to nature, appeal to tradition, false dilemma, appeal to majority, hasty generalization, appeal to authority

### Download Instructions
```bash
git clone https://github.com/Crowd-AI-Lab/cocolofa.git
cp cocolofa/*.json datasets/cocolofa/
```

### Label Distribution (train)
- none: 2,202
- slippery slope: 431
- appeal to worse problems: 421
- appeal to nature: 412
- appeal to tradition: 401
- false dilemma: 391
- appeal to majority: 383
- hasty generalization: 379
- appeal to authority: 350

---

## Dataset 5: BIG-Bench Mistake (Tyen et al., 2024)

### Overview
- **Source**: GitHub (WHGTyen/BIG-Bench-Mistake)
- **Size**: 2,186 CoT traces across 5 tasks
- **Format**: JSONL (one file per task)
- **Task**: Mistake location identification in CoT reasoning traces
- **Tasks**: word sorting (300), tracking shuffled objects (300), logical deduction (300), multistep arithmetic (300), Dyck languages (986)

### Download Instructions
```bash
git clone https://github.com/WHGTyen/BIG-Bench-Mistake.git
cp BIG-Bench-Mistake/*.jsonl datasets/bigbench_mistake/
```

---

## Dataset 6: RuozhiBench (Zhai et al., 2025)

### Overview
- **Source**: GitHub (LibrAIResearch/ruozhibench)
- **Size**: 675 bilingual (Chinese/English) questions
- **Format**: JSONL (multiple-choice and generative)
- **Task**: Detecting logical fallacies and misleading premises
- **Categories**: 16 categories including Logical Error, Commonsense Misunderstanding, Erroneous Assumption, Absurd Imagination, Scientific Misconception

### Download Instructions
```bash
git clone https://github.com/LibrAIResearch/ruozhibench.git
cp ruozhibench/data/*.jsonl datasets/ruozhibench/
```

---

## Dataset 7: MMLU Logical Fallacies

### Overview
- **Source**: HuggingFace (brucewlee1/mmlu-logical-fallacies)
- **Size**: 162 test + 4 validation questions
- **Format**: HuggingFace Dataset (multiple choice)
- **Task**: Multiple-choice fallacy identification

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("brucewlee1/mmlu-logical-fallacies")
dataset.save_to_disk("datasets/mmlu_logical_fallacies")
```

---

## Dataset 8: LogicClimate (Jin et al., 2022)

### Overview
- **Source**: GitHub (causalNLP/logical-fallacy)
- **Size**: Climate change-specific fallacy examples
- **Format**: CSV
- **Task**: Fallacy classification in climate change claims

### Download Instructions
```bash
git clone https://github.com/causalNLP/logical-fallacy.git
cp logical-fallacy/data/climate_*.csv datasets/logic_climate/
```

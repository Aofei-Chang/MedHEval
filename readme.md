# MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models

Welcome to the official repository for **MedHEval**, a benchmark designed to systematically evaluate hallucinations and mitigation strategies in Medical Large Vision-Language Models (Med-LVLMs).

---

## Overview

Medical Large Vision-Language Models (Med-LVLMs) offer great promise in clinical AI by combining image understanding and language generation. However, they frequently generate **hallucinations**—plausible but ungrounded or incorrect outputs—which can undermine trust and safety in medical applications.

**MedHEval** addresses this challenge by introducing a comprehensive benchmark to:
- Categorize hallucinations in Med-LVLMs,
- Evaluate model behavior across hallucination types,
- Compare mitigation strategies on multiple Med-LVLMs.

---

## Code Structure

MedHEval provides a modular and extensible codebase. Here's a high-level breakdown of the repository:

```
MedHEval/
│
├── benchmark_data/                  # Benchmark VQA and report data (excluding raw images)
│
├── code/
│   ├── baselines/
│   │   ├── (Med)-LVLMs/             # Inference code for baseline LVLMs (e.g., LLaVA, LLM-CXR, etc.)
│   │   └── mitigation/             # Implementations of hallucination mitigation strategies
│   │
│   ├── data_generation/            # Scripts to generate benchmark data split by dataset and hallucination type
│   │
│   └── evaluation/
│       ├── close_ended_evaluation/ # Evaluation pipeline for close-ended tasks (all hallucination types)
│       ├── open_ended_evaluation/  # Knowledge hallucination (open-ended)
│       └── report_eval/            # Visual hallucination evaluation from generated reports
│
├── scripts/                        # Setup and utility scripts
└── README.md
```

Each component includes its own README with detailed instructions.

---

## Hallucination Categories

MedHEval classifies hallucinations into three interpretable types:

1. **Visual Misinterpretation**  
   Misunderstanding or inaccurate reading of visual input (e.g., identifying a fracture that doesn’t exist).

2. **Knowledge Deficiency**  
   Errors stemming from gaps or inaccuracies in medical knowledge (e.g., incorrect drug-disease associations).

3. **Context Misalignment**  
   Failure to align visual understanding with textual prompts (e.g., answering unrelated to the question).

---

## Benchmark Components

MedHEval consists of the following key components:

- **📊 Diverse Medical VQA Datasets and Fine-Grained Metrics**  
  Includes both **close-ended** (yes/no, multiple choice) and **open-ended** (free-text, report generation) tasks. The benchmark provides structured metrics for each hallucination category.

- **🧠 Comprehensive Evaluation on Diverse (Med)-LVLMs**  
  MedHEval supports a broad range of models, including:
  - Generalist models (e.g., LLaVA, MiniGPT-4)
  - Medical-domain models (e.g., LLaVA-Med, LLM-CXR, CheXagent)
  - Retrieval-augmented or hybrid architectures

- **🛠️ Evaluation of Hallucination Mitigation Strategies**  
  Benchmarked techniques include:
  - Vision-text alignment tuning
  - Retrieval-augmented generation
  - Confidence calibration
  - Prompt-based and ensemble approaches

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MedHEval.git
cd MedHEval
```

### 2. Install Dependencies

#### Baseline Models Setup
Each Med-LVLM has its own requirements. Follow the official or customized instructions for environment setup and model checkpoints:

- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0)
- [LLaVA-Med-1.5](https://github.com/microsoft/LLaVA-Med)
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLM-CXR](https://github.com/hyn2028/llm-cxr)
- [CheXagent](https://github.com/Stanford-AIMI/CheXagent) *(use via HuggingFace Transformers)*
- [RadFM](https://github.com/chaoyi-wu/RadFM)
- [XrayGPT](https://github.com/mbzuai-oryx/XrayGPT) *(MiniGPT-4 environment)*

Each model’s folder under `code/baselines/(Med)-LVLMs/` contains:
- Inference scripts
- Config files
- Notes for modified packages (e.g., `transformers`)

#### Evaluation Modules

- **Close-ended evaluation**: Lightweight and model-agnostic  
  → [`code/evaluation/close_ended_evaluation`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/close_ended_evaluation)

- **Open-ended (Report)**: Requires older Python/tooling  
  → [`code/evaluation/report_eval`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/report_eval)

- **Open-ended (Knowledge)**: Lightweight (just needs `langchain`, `pydantic`, and LLM API access)  
  → [`code/evaluation/open_ended_evaluation`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/open_ended_evaluation)

---

## Data Access

The `benchmark_data/` folder includes annotation and split files.  
**Note**: Due to license restrictions, image data must be downloaded separately:

- [Slake](https://www.med-vqa.com/slake/)
- [VQA-RAD](https://osf.io/89kps/files/osfstorage)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

---

## Citation

If you use MedHEval in your research, please cite:

```bibtex
@article{chang2025medheval,
  title={MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models},
  author={Chang, Aofei and Huang, Le and Bhatia, Parminder and Kass-Hout, Taha and Ma, Fenglong and Xiao, Cao},
  journal={arXiv preprint arXiv:2503.02157},
  year={2025}
}
```
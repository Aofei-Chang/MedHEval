# 💭🤖 MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models 🏥📊

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
└── README.md
```

Each component may include its own README with detailed instructions.

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

- **🛠️ Evaluation of Hallucination Mitigation Strategies**  
  Benchmarked techniques include diverse mitigation methods focusing on visual bias or LLM bias:
  - [**OPERA**](https://github.com/shikiw/OPERA)  
  - [**VCD**](https://github.com/DAMO-NLP-SG/VCD)  
  - [**DoLa**](https://github.com/voidism/DoLa)  
  - [**AVISC**](https://github.com/sangminwoo/AvisC)  
  - **M3ID**  [M3ID Paper](https://arxiv.org/abs/2403.14003)
    _(No official code released; implementation follows AVISC.)_  
  - **DAMRO**  
    _(No official code released; implemented by ourself based on the paper: [DAMRO Paper](https://arxiv.org/abs/2410.04514))_  
  - [**PAI**](https://github.com/LALBJ/PAI)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MedHEval.git
cd MedHEval
```

### 2. Install Dependencies

MedHEval supports a **modular** and **flexible** setup. You can selectively evaluate any Med-LVLM and any hallucination type.


#### Baseline Models Setup

Please refer to the links below for the environment setup corresponding to the model you wish to evaluate. Some baseline models, such as LLM-CXR, may involve complex installation steps. In such cases, we recommend setting up the environment according to the official instructions and downloading the required model checkpoints. Once the environment is ready, you can run inference using our implementation provided under `./code/baselines/(Med)-LVLMs/`.

Each Med-LVLM has specific requirements. Please follow the official or customized setup instructions accordingly, including environment configuration and model checkpoint preparation.

- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med/tree/v1.0.0)
- [LLaVA-Med-1.5](https://github.com/microsoft/LLaVA-Med) -- We provide our customized environment setup and implementation, which includes modifications to the transformers package for hallucination mitigation baselines. Please refer to the corresponding folder [`code/baselines/(Med)-LVLMs/llava-med-1.5`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/baselines/Med-LVLMs/llava-med-1.5) for detailed instructions.
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
- [LLM-CXR](https://github.com/hyn2028/llm-cxr)
- [CheXagent](https://github.com/Stanford-AIMI/CheXagent) *(Note: use via HuggingFace Transformers)* This model can be loaded directly via Hugging Face transformers as no model and training codes are provided in their official repo. No special environment setup is needed beyond installing torch and transformers. We use the same environment as LLaVA-Med-1.5.
- [RadFM](https://github.com/chaoyi-wu/RadFM)
- [XrayGPT](https://github.com/mbzuai-oryx/XrayGPT) *(Note: MiniGPT-4 environment)* This model shares the same structure and environment setup as MiniGPT-4.

Each model’s folder under `code/baselines/(Med)-LVLMs/` contains:
- Inference scripts
- Config files
- Notes for modified packages (e.g., `transformers`)

#### Evaluation Modules
Note: The setup is separate from the (Med)-LVLMs and does not require installing the full environment of any specific Med-LVLM. Refer to [`code/evaluation`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation) for how to set up environments and how to run the evaluations.

- **Close-ended evaluation**: Lightweight, easy to use and model-agnostic  
  → [`code/evaluation/close_ended_evaluation`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/close_ended_evaluation)

- **Open-ended (Report)**: This setup is more involved, as some tools depend on older versions of Python and other packages.  
  → [`code/evaluation/report_eval`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/report_eval)

- **Open-ended (Knowledge)**: This setup is straightforward. Required packages: `langchain`, `pydantic`, and access to an LLM (e.g., Claude 3.5 Sonnet used in our experiments).  
  → [`code/evaluation/open_ended_evaluation`](https://github.com/Aofei-Chang/MedHEval/tree/main/code/evaluation/open_ended_evaluation)

---

## Data Access

The `benchmark_data/` folder includes annotation and split files.  
**Note**: Due to license restrictions, image data must be downloaded separately:

- [Slake](https://www.med-vqa.com/slake/)
- [VQA-RAD](https://osf.io/89kps/files/osfstorage)
- [IU-Xray](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view), as provided by [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT)
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
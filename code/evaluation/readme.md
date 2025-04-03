
# 🧪 Evaluation Environment & Code Overview

This folder provides all the code and instructions needed to evaluate **three types of hallucinations** in medical vision-language models. We support both **close-ended** and **open-ended** evaluation formats.

---

## ⚙️ Environment Setup

### ✅ 1. Close-ended Evaluation

Close-ended evaluation is simple to set up. Just ensure the following Python packages are installed:

```bash
pip install difflib numpy json tqdm
```

You can use any modern Python environment (e.g., Python 3.8–3.11).

---

### 🔄 2. Open-ended Evaluation

Open-ended evaluation for Type 1 hallucination (report evaluation) involves **two phases**, requiring two separate Python environments due to version conflicts:

#### 🔹 Phase 1: Report Evaluation Metrics (Legacy Dependencies)

1. Set up an environment with **Python 3.7**.
2. Install the required packages:

   ```bash
   pip install -r requirements_report.txt
   ```

3. Then you can run the evaluation script for **CheXbert** and **RadGraph**:

   ```bash
   python ./report_eval/run_eval.py
   ```

#### 🔹 Phase 2: RaTEScore & Other Modern Metrics (Newer Dependencies)

1. Set up a **second Python environment** (e.g., Python 3.9, 3.10).
2. Install the packages for RaTE and other newer metrics:

   ```bash
   pip install -r requirements_new_metrics.txt
   ```

3. Then you can run the newer metrics script to get **final evaluation results** including the intermediate results obtained from the Phrase 1:

   ```bash
   python ./report_eval/run_all_metrics.py
   ```

---

## 🧠 Evaluation Code Overview

We evaluate **three types of hallucinations**:

---

### 🎨 Type 1: Visual Misinterpretation Hallucination

#### 🔸 Close-ended (Classification):

- **Code**: `./report_eval/eval_type1_batch.py`
- **Script**: `./scripts/Visual-Open-ended/chair_eval.sh`  
  *(Edit paths to `test_data` and `inference_results` in the script as needed.)*

#### 🔸 Open-ended (Report Generation):

- **Report Generation Metrics**:
  1. Run `./report_eval/run_eval.py` (CheXbert, RadGraph)
  2. Then run `./report_eval/run_all_metrics.py` (RaTEScore, BertScore, etc.)
- **Script**: `./scripts/Visual-Open-ended/report_eval.sh`

- **Hallucination & Recall Metrics** (for key chest X-ray findings):
  - **Code**: `./report_eval/run_chair.py`
  - **Script**: `./scripts/Visual-Open-ended/chair_eval.sh`

---

### 📚 Type 2: Knowledge Deficiency Hallucination

#### 🔸 Close-ended:

- **Code**: `./close_ended_evaluation/eval_type23_close.py`

#### 🔸 Open-ended:

- **LLM-based Evaluation (Claude 3.5 Sonnet)**:
  - **Code**: `./open_ended_evaluation/knowledge_hallucination_score.py`

- **Other Generation Metrics**:
  - **Code**: `./open_ended_evaluation/generation_metrics.py`

---

### 🧩 Type 3: Context Misalignment Hallucination

#### 🔸 Close-ended:

- **Code**: `./close_ended_evaluation/eval_type23_close.py`

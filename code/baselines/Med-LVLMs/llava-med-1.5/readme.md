
### 🔧 Model Setup: Installing Required Packages

```bash
# Create and activate a new Conda environment
conda create -n llava_med_v1.5 python=3.10 -y
conda activate llava_med_v1.5

# Upgrade pip to enable PEP 660 support
pip install --upgrade pip

# Install the current project in editable mode
pip install -e .
```

The core implementation of our hallucination mitigation baselines can be found in:

```
transformers-4.37.2/src/transformers/generation/utils.py
```

To install the modified `transformers` package (v4.37.2) in editable mode:

```bash
python -m pip install -e transformers-4.37.2
```
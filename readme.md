## Baselines

- All implementations strictly adhere to the original settings of the respective baselines.
- For more details, please refer to the official implementations via the links provided in our manuscript, as outlined in Appendix A.
- We include the original code for all evaluated LVLMs along with the corresponding inference scripts:
  - Available in the folder: `code/baselines/LVLMs/`
  - All model inference scripts can be found under the `inference_scripts` subfolder within each LVLM's directory.

------

## Evaluation

We provide the code for open-ended evaluation using GPT-4:

- Located in the folder: `code/evaluation`

Close-ended evaluations are conducted using existing benchmarks, which can be accessed here: [CARES Evaluation](https://github.com/richard-peng-xia/CARES/tree/main/src/eval)

------

## Data Generation

For data generation, the code is organized by data sources and hallucination types, as each source and type require distinct processing methods.

- Please find the relevant code in the folder: `code/data_generation

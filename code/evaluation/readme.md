### The evaluation codes for all three types of hallucinations

1.Close-ended Visual Misinterpretation Hallucination:

- code: ./report_eval/eval_type1_batch.py
- scripts: ./scripts/Visual-Open-ended/chair_eval.sh

2.Open-ended Visual Misinterpretation Hallucination (report evaluation):

- Metrics for report generation ()
  - code: (1) firstly run ./report_eval/run_eval.py, to firstly run the RadGraph and CheXbert; (2) then run RateScore and other generation metrics. 

- Metrics for hallucination and recall rate (In the range of key findings in chest X-rays)
  - code: ./report_eval/run_chair.py
  - scripts: ./scripts/Visual-Open-ended/chair_eval.sh

3.Close-ended Knowledge deficiency Hallucination:

- code: eval_type23_close.py

4.Open-ended Knowledge deficiency Hallucination:

- We utilize Claude 3.5 Sonnet for a open-ended evaluation
  - code: open_ended_evaluation/knowledge_hallucination_score.py
- Other generation metrics: 
  - ./open_ended_evaluation/generation_metrics.py

5.Close-ended Context Misalignment Hallucination:

- code: ./close_ended_evaluation/eval_type23_close.py
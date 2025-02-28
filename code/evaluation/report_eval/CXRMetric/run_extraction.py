import json
import numpy as np
import os
import re
import pandas as pd
import torch
current_working_directory = os.getcwd()
print("Current Working Directory:", current_working_directory)
import sys
sys.path.append("/data/aofei/hallucination/mitigation/report_eval")
# from CXRMetric.radgraph_evaluate_model import run_radgraph
import config as config

"""Computes 4 individual metrics and a composite metric on radiology reports."""


CHEXBERT_PATH = config.CHEXBERT_PATH
RADGRAPH_PATH = config.RADGRAPH_PATH

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

NORMALIZER_PATH = "CXRMetric/normalizer.pkl"
COMPOSITE_METRIC_V0_PATH = "CXRMetric/composite_metric_model.pkl"
COMPOSITE_METRIC_V1_PATH = "CXRMetric/radcliq-v1.pkl"

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]


# cache_path = "cache/"
# cache_path = "/data/aofei/hallucination/mitigation/report_eval/cache"
cache_path = "/data/aofei/hallucination/mitigation/report_eval/cache_chexpert"
pred_label_path = os.path.join(cache_path, "pred_labels/")
gt_label_path = os.path.join(cache_path, "gt_labels/")
# pred_label_path = os.path.join(cache_path, "pred_labels_chexpert.csv")
# gt_label_path = os.path.join(cache_path, "gt_labels_chexpert.csv")
weights = {"bigram": (1/2., 1/2.)}
composite_metric_col_v0 = "RadCliQ-v0"
composite_metric_col_v1 = "RadCliQ-v1"


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

import pandas as pd

def calculate_metrics(inference_csv, ground_truth_csv):
    # Load CSV files
    df_infer = pd.read_csv(inference_csv)
    df_gt = pd.read_csv(ground_truth_csv)
    
    # Ensure both CSVs have the same number of rows
    assert len(df_infer) == len(df_gt), "The two CSV files must have the same number of rows"
    
    # Get symptom columns (excluding 'report' column)
    symptom_columns = [col for col in df_infer.columns if col != "report"]
    
    hallucinated_count = 0
    total_generated_count = 0
    true_positive_count = 0
    false_negative_count = 0
    true_negative_count = 0
    false_positive_count = 0
    total_positive_gt = 0
    
    for symptom in symptom_columns:
        # Get the inferred and ground truth labels for the current symptom
        infer_labels = df_infer[symptom].fillna("")
        gt_labels = df_gt[symptom].fillna("")
        
        # Convert non-empty values to numeric (-1, 0, 1) and ignore empty values
        infer_labels = pd.to_numeric(infer_labels, errors='coerce')
        gt_labels = pd.to_numeric(gt_labels, errors='coerce')
        
        # Filter out cases where ground truth is uncertain (-1)
        valid_indices = gt_labels != -1
        infer_labels = infer_labels[valid_indices]
        gt_labels = gt_labels[valid_indices]
        
        # Compute metrics
        tp = ((infer_labels == 1) & (gt_labels == 1)).sum()
        tn = ((infer_labels == 0) & (gt_labels == 0)).sum()
        fp = ((infer_labels == 1) & (gt_labels == 0)).sum()
        fn = ((infer_labels == 0) & (gt_labels == 1)).sum()
        
        hallucinated_count += fp + fn
        total_generated_count += ((infer_labels != -1) & (infer_labels.notna())).sum()
        true_positive_count += tp
        false_negative_count += fn
        true_negative_count += tn
        false_positive_count += fp
        total_positive_gt += (gt_labels == 1).sum()
    
    # Calculate hallucination score
    hallucination_score = hallucinated_count / total_generated_count if total_generated_count > 0 else 0
    
    # Calculate recall (sensitivity)
    recall = true_positive_count / total_positive_gt if total_positive_gt > 0 else 0
    
    # Calculate specificity
    specificity = true_negative_count / (true_negative_count + false_positive_count) if (true_negative_count + false_positive_count) > 0 else 0
    
    # Calculate accuracy
    accuracy = (true_positive_count + true_negative_count) / (true_positive_count + true_negative_count + false_positive_count + false_negative_count) if (true_positive_count + true_negative_count + false_positive_count + false_negative_count) > 0 else 0
    
    return hallucination_score, recall, specificity, accuracy

def calculate_hallucination_score_old(inference_csv, ground_truth_csv):
    # Load CSV files
    df_infer = pd.read_csv(inference_csv)
    df_gt = pd.read_csv(ground_truth_csv)
    
    # Ensure both CSVs have the same number of rows
    assert len(df_infer) == len(df_gt), "The two CSV files must have the same number of rows"
    
    # Get symptom columns (excluding 'report' column)
    symptom_columns = [col for col in df_infer.columns if col != "report"]
    
    hallucinated_count = 0
    total_generated_count = 0
    
    for symptom in symptom_columns:
        # Get the inferred and ground truth labels for the current symptom
        print(symptom)
        infer_labels = df_infer[symptom].fillna("")
        gt_labels = df_gt[symptom].fillna("")
        
        # Convert non-empty values to numeric (-1, 0, 1) and ignore empty values
        infer_labels = pd.to_numeric(infer_labels, errors='coerce')
        gt_labels = pd.to_numeric(gt_labels, errors='coerce')
        print(infer_labels, gt_labels,"labaless")
        # Identify hallucinated symptoms (1 in inference but 0 in ground truth)
        # hallucinated = (infer_labels == 1) & (gt_labels == 0)
        hallucinated = ((infer_labels == 1) & (gt_labels == 0)) | ((infer_labels == 0) & (gt_labels == 1))
        
        # Count hallucinated symptoms
        hallucinated_count += hallucinated.sum()
        
        # Count all generated symptoms (1, -1, or 0 in inference but not empty)
        total_generated_count += infer_labels.notna().sum()
    print(hallucinated_count, total_generated_count, "counts")
    # Calculate hallucination score
    hallucination_score = hallucinated_count / total_generated_count if total_generated_count > 0 else 0
    
    return hallucination_score

# def add_semb_col(pred_df, semb_path, gt_path):
#     """Computes s_emb and adds scores as a column to prediction df."""
#     label_embeds = torch.load(gt_path)
#     pred_embeds = torch.load(semb_path)
#     list_label_embeds = []
#     list_pred_embeds = []
#     for data_idx in sorted(label_embeds.keys()):
#         list_label_embeds.append(label_embeds[data_idx])
#         list_pred_embeds.append(pred_embeds[data_idx])
#     np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
#     np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
#     scores = []
#     for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
#         sim_scores = (label * pred).sum() / (
#             np.linalg.norm(label) * np.linalg.norm(pred))
#         scores.append(sim_scores)
#     pred_df["semb_score"] = scores
#     return pred_df

def calc_metric(gt_csv, pred_csv, out_csv, use_idf): # TODO: support single metrics at a time
    """Computes four metrics and composite metric scores."""
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    cache_gt_csv = os.path.join(
        os.path.dirname(gt_csv), f"cache_{os.path.basename(gt_csv)}")
    cache_pred_csv = os.path.join(
        os.path.dirname(pred_csv), f"cache_{os.path.basename(pred_csv)}")
    gt = pd.read_csv(gt_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)
    pred = pd.read_csv(pred_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)

    # Keep intersection of study IDs
    gt_study_ids = set(gt[STUDY_ID_COL_NAME])
    pred_study_ids = set(pred[STUDY_ID_COL_NAME])
    shared_study_ids = gt_study_ids.intersection(pred_study_ids)
    print(f"Number of shared study IDs: {len(shared_study_ids)}")
    gt = gt.loc[gt[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()
    pred = pred.loc[pred[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()

    gt.to_csv(cache_gt_csv)
    pred.to_csv(cache_pred_csv)

    # check that length and study IDs are the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)
    assert (gt[STUDY_ID_COL_NAME].equals(pred[STUDY_ID_COL_NAME]))
    
    # run encode.py to make the semb column
    os.system(f"mkdir -p {cache_path}")
    os.system(f"python /data/aofei/hallucination/mitigation/report_eval/CXRMetric/CheXbert/src/label.py -c {CHEXBERT_PATH} -d {cache_pred_csv} -o {pred_label_path}")
    os.system(f"python /data/aofei/hallucination/mitigation/report_eval/CXRMetric/CheXbert/src/label.py -c {CHEXBERT_PATH} -d {cache_gt_csv} -o {gt_label_path}")

    print("finish labeling")
    # pred = add_semb_col(pred, pred_label_path, gt_label_path)
    pred_label_file = os.path.join(pred_label_path, "labeled_reports.csv")
    gt_label_file = os.path.join(gt_label_path, "labeled_reports.csv")
    # chair_score = calculate_metrics(pred_label_file, gt_label_file)
    hallucination_score, recall, specificity, accuracy = calculate_metrics(pred_label_file, gt_label_file)
    # print(chair_score)
    with open(out_csv, "w") as f:
        f.write("CHAIR: " + str(hallucination_score) + "\n")
        f.write("Recall: " + str(recall) + "\n")
        f.write("Specificity: " + str(specificity) + "\n")
        f.write("Accuracy: " + str(accuracy) + "\n")
        f.close()
    # pred.to_csv(out_csv)

import json
import numpy as np
import os
import re
import pandas as pd
import pickle
import torch

from bert_score import BERTScorer
from fast_bleu import BLEU
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

current_working_directory = os.getcwd()
print("Current Working Directory:", current_working_directory)
import sys
sys.path.append("/data/aofei/hallucination/mitigation/report_eval")
from CXRMetric.radgraph_evaluate_model import run_radgraph
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
cache_path = "/data/aofei/hallucination/mitigation/report_eval/cache"
pred_embed_path = os.path.join(cache_path, "pred_embeddings.pt")
gt_embed_path = os.path.join(cache_path, "gt_embeddings.pt")
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

def add_bleu_col(gt_df, pred_df):
    """Computes BLEU-2 and adds scores as a column to prediction df."""
    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        predicted_report = \
            prep_reports([pred_row[REPORT_COL_NAME].values[0]])[0]
        if len(pred_row) == 1:
            bleu = BLEU([gt_report], weights)
            score = bleu.get_score([predicted_report])["bigram"]
            assert len(score) == 1
            _index = pred_df.index[
                pred_df[STUDY_ID_COL_NAME]==row[STUDY_ID_COL_NAME]].tolist()[0]
            pred_df.at[_index, "bleu_score"] = score[0]
    return pred_df

def add_bertscore_col(gt_df, pred_df, use_idf):
    """Computes BERTScore and adds scores as a column to prediction df."""
    test_reports = gt_df[REPORT_COL_NAME].tolist()
    test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
    method_reports = pred_df[REPORT_COL_NAME].tolist()
    method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
        idf=use_idf,
        idf_sents=test_reports)
    _, _, f1 = scorer.score(method_reports, test_reports)
    pred_df["bertscore"] = f1
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    list_label_embeds = []
    list_pred_embeds = []
    for data_idx in sorted(label_embeds.keys()):
        list_label_embeds.append(label_embeds[data_idx])
        list_pred_embeds.append(pred_embeds[data_idx])
    np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
    np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    pred_df["semb_score"] = scores
    return pred_df

def add_radgraph_col(pred_df, entities_path, relations_path):
    """Computes RadGraph F1 and adds scores as a column to prediction df."""
    study_id_to_radgraph = {}
    with open(entities_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] = float(f1)
            except:
                continue
    with open(relations_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] += float(f1)
                study_id_to_radgraph[int(study_id)] /= float(2)
            except:
                continue
    radgraph_scores = []
    count = 0
    for i, row in pred_df.iterrows():
        radgraph_scores.append(study_id_to_radgraph[int(row[STUDY_ID_COL_NAME])])
    pred_df["radgraph_combined"] = radgraph_scores
    return pred_df

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

    # add blue column to the eval df
    # pred = add_bleu_col(gt, pred)

    # add bertscore column to the eval df
    # pred = add_bertscore_col(gt, pred, use_idf)

    # run encode.py to make the semb column
    os.system(f"mkdir -p {cache_path}")
    os.system(f"python /data/aofei/hallucination/mitigation/report_eval/CXRMetric/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {cache_pred_csv} -o {pred_embed_path}")
    os.system(f"python /data/aofei/hallucination/mitigation/report_eval/CXRMetric/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {cache_gt_csv} -o {gt_embed_path}")
    pred = add_semb_col(pred, pred_embed_path, gt_embed_path)
# 
    # run radgraph to create that column
    entities_path = os.path.join(cache_path, "entities_cache.json")
    relations_path = os.path.join(cache_path, "relations_cache.json")
    run_radgraph(cache_gt_csv, cache_pred_csv, cache_path, RADGRAPH_PATH,
                 entities_path, relations_path)
    pred = add_radgraph_col(pred, entities_path, relations_path)

    # compute composite metric: RadCliQ-v0
    # with open(COMPOSITE_METRIC_V0_PATH, "rb") as f:
    #     composite_metric_v0_model = pickle.load(f)
    # with open(NORMALIZER_PATH, "rb") as f:
    #     normalizer = pickle.load(f)
    # # normalize
    # input_data = np.array(pred[COLS])
    # norm_input_data = normalizer.transform(input_data)
    # # generate new col
    # radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
    # pred[composite_metric_col_v0] = radcliq_v0_scores

    # # compute composite metric: RadCliQ-v1
    # with open(COMPOSITE_METRIC_V1_PATH, "rb") as f:
    #     composite_metric_v1_model = pickle.load(f)
    # input_data = np.array(pred[COLS])
    # radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
    # pred[composite_metric_col_v1] = radcliq_v1_scores

    # save results in the out folder
    pred.to_csv(out_csv)

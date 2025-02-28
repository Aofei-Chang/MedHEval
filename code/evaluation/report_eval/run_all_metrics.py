# from datasets import load_metric
from bert_score import score as bert_score
import nltk
nltk.download('punkt')
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
from tabulate import tabulate

from RaTEScore import RaTEScore
ratescore = RaTEScore()

nltk.download('punkt_tab')
nltk.download('wordnet')

def eval_pairs(model_answers, ground_truths, eval_res_file, RadGraphFile):

    # Initialize metrics
    bleu_scores = []
    meteor_scores = []
    rouge_scores = {
        "rouge1": {"precision": [], "recall": [], "f1": []},
        "rouge2": {"precision": [], "recall": [], "f1": []},
        "rougeL": {"precision": [], "recall": [], "f1": []}
    }
    bert_scores = {
        "precision": [],
        "recall": [],
        "f1": []
    }

    # BertScore
    print("calculating bertscore")
    P, R, F1 = bert_score(model_answers, ground_truths, lang='en', verbose=True)
    bert_scores['precision'] = P.tolist()
    bert_scores['recall'] = R.tolist()
    bert_scores['f1'] = F1.tolist()

    # METEOR, ROUGE, and BLEU calculations
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_fn = SmoothingFunction().method4

    for model_answer, ground_truth in zip(model_answers, ground_truths):
        # Tokenize the texts
        tokenized_model_answer = nltk.word_tokenize(model_answer)
        tokenized_ground_truth = nltk.word_tokenize(ground_truth)

        # BLEU
        reference = [tokenized_ground_truth]
        candidate = tokenized_model_answer
        bleu_score_val = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
        bleu_scores.append(bleu_score_val)

        # METEOR
        meteor_score_val = meteor_score([tokenized_ground_truth], tokenized_model_answer)
        meteor_scores.append(meteor_score_val)

        # ROUGE
        rouge_score_val = rouge.score(model_answer, ground_truth)
        for key in rouge_scores:
            rouge_scores[key]["precision"].append(rouge_score_val[key].precision)
            rouge_scores[key]["recall"].append(rouge_score_val[key].recall)
            rouge_scores[key]["f1"].append(rouge_score_val[key].fmeasure)

    # Calculate average results
    average_results = {
        "BERTScore": {
            "precision": np.mean(bert_scores['precision']),
            "recall": np.mean(bert_scores['recall']),
            "f1": np.mean(bert_scores['f1'])
        },
        "BLEU": np.mean(bleu_scores),
        "METEOR": np.mean(meteor_scores),
        "ROUGE-1": {
            "precision": np.mean(rouge_scores['rouge1']['precision']),
            "recall": np.mean(rouge_scores['rouge1']['recall']),
            "f1": np.mean(rouge_scores['rouge1']['f1'])
        },
        "ROUGE-2": {
            "precision": np.mean(rouge_scores['rouge2']['precision']),
            "recall": np.mean(rouge_scores['rouge2']['recall']),
            "f1": np.mean(rouge_scores['rouge2']['f1'])
        },
        "ROUGE-L": {
            "precision": np.mean(rouge_scores['rougeL']['precision']),
            "recall": np.mean(rouge_scores['rougeL']['recall']),
            "f1": np.mean(rouge_scores['rougeL']['f1'])
        }
    }
    
    ratescore_results = ratescore.compute_score(model_answers, ground_truths)
    average_results["RaTEScore"] = np.mean(ratescore_results)
    # average_results["RaTEScore"] = 0

    # Load RadGraph file
    radgraph_data = pd.read_csv(RadGraphFile)

    # Calculate average values for semb_score and radgraph_combined
    average_semb_score = radgraph_data['semb_score'].mean()
    average_radgraph_combined = radgraph_data['radgraph_combined'].mean()

    # Add these averages to the results
    average_results["semb_score"] = average_semb_score
    average_results["radgraph_combined"] = average_radgraph_combined

    table = [
        ["Metric", "Precision", "Recall", "F1"],
        ["BERTScore", 
            "{:.4f}".format(average_results["BERTScore"]["precision"]),
            "{:.4f}".format(average_results["BERTScore"]["recall"]),
            "{:.4f}".format(average_results["BERTScore"]["f1"])],
        ["BLEU", "-", "-", "{:.4f}".format(average_results["BLEU"])],
        ["METEOR", "-", "-", "{:.4f}".format(average_results["METEOR"])],
        ["ROUGE-1", 
            "{:.4f}".format(average_results["ROUGE-1"]["precision"]),
            "{:.4f}".format(average_results["ROUGE-1"]["recall"]),
            "{:.4f}".format(average_results["ROUGE-1"]["f1"])],
        ["ROUGE-2", 
            "{:.4f}".format(average_results["ROUGE-2"]["precision"]),
            "{:.4f}".format(average_results["ROUGE-2"]["recall"]),
            "{:.4f}".format(average_results["ROUGE-2"]["f1"])],
        ["ROUGE-L", 
            "{:.4f}".format(average_results["ROUGE-L"]["precision"]),
            "{:.4f}".format(average_results["ROUGE-L"]["recall"]),
            "{:.4f}".format(average_results["ROUGE-L"]["f1"])],
        ["RaTEScore", "-", "-", "{:.4f}".format(average_results["RaTEScore"])],
        ["semb_score", "-", "-", "{:.4f}".format(average_results["semb_score"])],
        ["radgraph_combined", "-", "-", "{:.4f}".format(average_results["radgraph_combined"])]
    ]

    # Write the table to the file
    with open(eval_res_file, "a") as file:
        file.write("\n" + tabulate(table, headers="firstrow", tablefmt="grid"))


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_answers_file', type=str, required=True, help='Path to the file with model answers')
    parser.add_argument('--RadGraphFile', type=str, required=True, help='Path to the file with evaluation results')
    parser.add_argument('--eval_res_file', type=str, required=True, help='Path to the file to write evaluation results')
    args = parser.parse_args()

    # Load model answers and ground truths
    # with open(args.model_answers, "r") as file:
    #     model_answers = file.readlines()
    with open(args.model_answers_file, 'r') as f:
        results = [json.loads(line) for line in f]
    answers, gts = [], []
    for i in results:
        if "text" in i:
            answers.append(i["text"].strip("Assistant:"))
        else:
            answers.append(i["model_answer"])
        if "ground_truth" in i:
            gts.append(i["ground_truth"])
        else:
            gts.append(i["gt"] if "gt" in i else i['gt_ans'])
    eval_pairs(answers, gts, args.eval_res_file, args.RadGraphFile)
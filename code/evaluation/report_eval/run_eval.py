import argparse
import json
import collections
import random
import difflib
import pandas as pd    
from nltk.translate.bleu_score import sentence_bleu
from tabulate import tabulate

import warnings
warnings.simplefilter('ignore')

def parse_option():
    parser = argparse.ArgumentParser('Evaluation for LLaVA Generated Outputs', add_help=False)
    parser.add_argument('--gt', type=str, default="test.json", help='path to groundtruth file', )
    parser.add_argument('--candidate', type=str, default="candidate.json", help='path to candidate answer file', )
    parser.add_argument('--pred', type=str, default="answer-file-llava-zeorshot.jsonl", help='path to prediction file', )
    parser.add_argument('--eval_res', type=str, default="eval_res.txt", help='path to prediction file', )
    parser.add_argument('--mc', type=bool, default=False, help='whether evaluate multi-choice type', )
    parser.add_argument('--report', type=bool, default=False, help='whether evaluate multi-choice type', )
    args, unparsed = parser.parse_known_args()
    return args

def load_jsonl(path):
    data=[]
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data 


def evaluate_report(gt, pred, out_file):
    from CXRMetric.run_eval import calc_metric
    calc_metric(gt_csv=gt, pred_csv=pred, out_csv=out_file, use_idf=False)

if __name__ == '__main__':
    args = parse_option()
    import csv
    import os
    dataset = args.gt.split("/")[-2]
    print(f"\n========\n {dataset}")

    if not args.report:
        gt = json.load(open(args.gt, 'r'))
        # candidate = json.load(open(args.candidate, 'r'))
        pred = load_jsonl(args.pred)

        gt_ids = [item['id'] for item in gt]
        pred_ids = [item['question_id'] for item in pred]
        num_gt_ids, num_pred_ids = len(gt_ids), len(pred_ids)
        print(f'num_gt_ids: {num_gt_ids} || num_pred_ids: {num_pred_ids}')
        # import pdb; pdb.set_trace()
        assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"
    if args.mc:
        # perform evaluation
        results = evaluate_mc(gt, pred)
        print(results)
        with open(args.eval_res, "w") as f:
            f.write(results)
            f.close()
        exit()
    elif args.report:
        # perform evaluation
        pred_path = args.pred.replace(".csv", ".jsonl")
        with open(pred_path, "r") as f:
            pred_data = [json.loads(line) for line in f]

            # Write test data to CSV
        with open(args.pred, 'w', newline='') as csvfile:
            fieldnames = ['study_id', 'report']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data in pred_data:
                new_line = dict()
                try:
                    new_line['study_id'] = int(data['question_id'])
                except ValueError:
                    new_line['study_id'] = int(data['question_id'].split("train")[-1])
                if "model_answer" in data:
                    data['text'] = data['model_answer']
                new_line['report'] = data['text'].strip("Assistant:")
                if len(new_line['report']) == 0:
                    new_line['report'] = data['text']
                if len(new_line['report']) < 10:
                    new_line['report'] = "No report generated."
                writer.writerow(new_line)
        if not os.path.exists(args.gt):
            gt_path = args.gt.replace(".csv", ".json")
            with open(gt_path, "r") as f:
                gt_data = json.load(f)
                # Write test data to CSV
            with open(args.gt, 'w', newline='') as csvfile:
                fieldnames = ['study_id', 'report']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for data in gt_data:
                    new_line = dict()
                    if "qid" in data:
                        data['id'] = data['qid']
                    try:
                        new_line['study_id'] = int(data['id'])
                    except ValueError:
                        new_line['study_id'] = int(data['id'].split("train")[-1])
                    if "conversation" in data:
                        new_line['report'] = data['conversations'][1]['value']
                    else:
                        new_line['report'] = data['answer']
                    writer.writerow(new_line)
        results = evaluate_report(args.gt, args.pred, args.eval_res)
        # print(results)
        exit()
    else:
        # perform evaluation
        results = evaluate(gt, pred, candidate=None)
        print(results)
        with open(args.eval_res, "w") as f:
            f.write(results)
            f.close()

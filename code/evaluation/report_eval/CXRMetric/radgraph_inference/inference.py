import math
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import argparse

"""Code adapted from https://physionet.org/content/radgraph/1.0.0: models/inference.py."""

def preprocess_reports(data_path, start, end, sentence=False, image=False):
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    impressions = pd.read_csv(data_path)
    if start != None and end != None:
        impressions = impressions.iloc[start:end]
    final_list = []
    for idx, row in impressions.iterrows():
        if (isinstance(row["report"], float) and math.isnan(row["report"])): continue
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row["report"]).split()
        temp_dict = {}

        if not sentence:  # Report-level
            if image:  # Different image can have different reports
                temp_dict["doc_key"] = f"{row['dicom_id']}_{row['study_id']}"
            else:
                temp_dict["doc_key"] = str(row["study_id"])
        else:  # Sentence-level
            temp_dict["doc_key"] = f"{row['study_id']}_{row['sentence_id']}"
        
        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

        if(idx % 1000 == 0):
            print(f"{idx+1} reports done")
    
    print(f"{idx+1} reports done")
    
    with open("/data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_input.json",'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")

def run_inference(model_path, cuda):
    
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path = "/data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_output.json"
    data_path = "/data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_input.json"
    # current_working_directory = os.getcwd()
    # print("Current Working alen Directory:", current_working_directory)
    
    os.system(f"allennlp predict {model_path} {data_path} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {-1} \
            --silent")

def postprocess_reports(data_source, data_split):
    
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    file_name = f"/data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_output.json"
    data = []

    with open(file_name,'r') as f:
        for line in f:
            data.append(json.loads(line))

    for file in data:
        postprocess_individual_report(file, final_dict, data_source=data_source, data_split=data_split)
    
    return final_dict

def postprocess_individual_report(file, final_dict, data_source=None, data_split="inference"):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = data_split

        if file['doc_key'] in final_dict:  # Handle duplicate study IDs.
            final_dict[file['doc_key'] + '+'] = temp_dict
        else:
            final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
def get_entity(n,r,s):
    
    """Gets the entities for individual reports
    
    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence
        
    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json 
    
    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
        for i,val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab,str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx+1)] = temp_dict
    
    return dict_entity

def cleanup():
    """Removes all the temporary files created during the inference process
    
    """
    # os.system("rm temp_file_list.json")
    os.system("rm /data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_input.json")
    os.system("rm /data/aofei/hallucination/mitigation/report_eval/cache/temp_dygie_output.json")

def _json_to_csv(path, csv_path):
    with open(path, "r") as f:
        match_results = json.load(f)
    reconstructed_reports = []
    for _, (_, train, match) in match_results.items():
        test_report_id = match[0][0][:8]
        reconstructed_reports.append((test_report_id, train))
    pd.DataFrame(reconstructed_reports, columns=["study_id", "report"]).to_csv(csv_path)

def _add_ids_column(
            csv_path, study_id_csv_path, output_path):
    with open(csv_path, "r") as f:
        generated_reports = pd.read_csv(f)
    with open(study_id_csv_path, "r") as f:
        ids_csv = pd.read_csv(f)
        study_ids = ids_csv["study_id"]
        dicom_ids = ids_csv["dicom_id"]
        subject_ids = ids_csv["subject_id"]
    generated_reports["study_id"] = study_ids
    generated_reports["dicom_id"] = dicom_ids
    generated_reports["subject_id"] = subject_ids
    #generated_reports.drop_duplicates(subset=["study_id"], keep="first")
    generated_reports.to_csv(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=False,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')

    
    args = parser.parse_args()
    
    run(args.model_path, args.data_path, args.out_path, args.cuda_device)

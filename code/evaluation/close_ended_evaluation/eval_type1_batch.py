import json
from utils.eval_yesno import evaluate_yes_no
from utils.eval_multichoice import eval_mc
import re
import numpy as np

from utils.type1_utils import eval_all

root_path = r"/data/aofei/hallucination"


if __name__ == "__main__":

    #load the original data
    slake_ori_path = f"{root_path}/Slake/slake_qa_pairs.json"
    with open(slake_ori_path, 'r') as file:
        slake_ori = json.load(file)
        
    rad_ori_path = f"{root_path}/VQA_RAD/rad_vqa_pairs.json"
    with open(rad_ori_path, 'r') as file:
        rad_ori = json.load(file)

    xray_ori_path = f"{root_path}/IU_Xray/xray_closed_pairs.json"
    with open(xray_ori_path, 'r') as file:
        xray_ori = json.load(file)
        
    slake_id_to_ori = dict()
    rad_id_to_ori = dict()
    xray_id_to_ori = dict()
    for i in slake_ori:
        slake_id_to_ori[i['qid']] = i
    for i in rad_ori:
        rad_id_to_ori[i['qid']] = i
    for i in xray_ori:
        xray_id_to_ori[i['qid']] = i

    mimic_ori_path = f"{root_path}/mimic_cxr/inference_close1/mimic_cxr_closed_pairs.json"
    with open(mimic_ori_path, 'r') as file:
        mimic_ori = json.load(file)
    mimic_id_to_ori = dict()    
    for i in mimic_ori:
        mimic_id_to_ori[i['qid']] = i

    baselines = ['original', 'DoLa', 'PAI', 'm3id', 'VCD', 'damro']
    # baselines = ['original', 'DoLa', 'PAI', 'm3id', 'VCD', 'avisc', 'damro']
    # baselines = ['original']
    # for baseline in baselines:
    #     slake_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6/Slake/pred_{baseline}.jsonl"
    #     rad_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6/VQA_RAD/pred_{baseline}.jsonl"
    #     xray_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6/IU_Xray/pred_{baseline}.jsonl"
    #     mimic_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6/mimic_cxr_closed/pred_{baseline}.jsonl"
    #     res = eval_all(slake_path, rad_path, xray_path, mimic_path, slake_ori, slake_id_to_ori, rad_ori, rad_id_to_ori, xray_ori, xray_id_to_ori, mimic_ori, mimic_id_to_ori)
    #     res_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6/{baseline}_res.txt"
    #     with open(res_path, 'w') as file:
    #         file.write(str(res))

    for baseline in baselines:
            slake_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6_13b/Slake/pred_{baseline}.jsonl"
            rad_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6_13b/VQA_RAD/pred_{baseline}.jsonl"
            xray_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6_13b/IU_Xray/pred_{baseline}.jsonl"
            mimic_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6_13b/mimic_cxr_closed/pred_{baseline}.jsonl"
            res = eval_all(slake_path, rad_path, xray_path, mimic_path, slake_ori, slake_id_to_ori, rad_ori, rad_id_to_ori, xray_ori, xray_id_to_ori, mimic_ori, mimic_id_to_ori)
            res_path = f"{root_path}/MedHEval/type1/baselines_llava_v1.6_13b/{baseline}_res.txt"
            with open(res_path, 'w') as file:
                file.write(str(res))
import json
from utils.eval_yesno import evaluate_yes_no

root_path = r"/data/aofei/hallucination"

def eval_yes_no(results):
    answers = [{"text":line['text']} for line in results]
    if "gt" in results[0]:
        labels = [line['gt'] for line in results]
    else:
        labels = [line['gt_ans'] for line in results]
    return evaluate_yes_no(answers, labels)

def eval_closed_single(ori_data, id_to_ori, inference_res):
    yn_ids, mc_ids = [], []
    for i in ori_data:
        if i['ground_truth_type'] == "binary":
            yn_ids.append(i['qid'])
        elif i['question_type'] == "multi-choice":
            mc_ids.append(i['qid'])
    
    yn_results, mc_results= [], []
    
    for i in inference_res:
        if i['question_id'] in yn_ids:
            yn_results.append(i)
            
        elif i['question_id'] in mc_ids:
            ori_i = id_to_ori[i['question_id']]
            i['choices'] = ori_i['choices']
            i['question_type'] = ori_i['question_type']
            mc_results.append(i)

    yn_acc_type = eval_yes_no(yn_results)
    
    return yn_acc_type


mimic_ori2_path = f"{root_path}/mimic_cxr/type2_close/mimic_cxr_closed_pairs.json"
with open(mimic_ori2_path, 'r') as file:
    mimic_ori2 = json.load(file)
mimic_id_to_ori2 = dict()    
for i in mimic_ori2:
    mimic_id_to_ori2[i['qid']] = i
    
mimic_ori3_path = f"{root_path}/mimic_cxr/type3/mimic_cxr_closed_pairs.json"
with open(mimic_ori3_path, 'r') as file:
    mimic_ori3 = json.load(file)
mimic_id_to_ori3 = dict()    
for i in mimic_ori3:
    mimic_id_to_ori3[i['qid']] = i

def eval_mimic_single(type2_infer_path, type3_infer_path):
    with open(type2_infer_path, 'r') as f:
        mimic_results = []
        for line in f:
            mimic_results.append(json.loads(line))
    
    slake_f_accs = eval_closed_single(mimic_ori2, mimic_id_to_ori2, mimic_results)
    
    with open(type3_infer_path, 'r') as f:
        mimic_results = []
        for line in f:
            mimic_results.append(json.loads(line))
    
    slake_f_accs3 = eval_closed_single(mimic_ori3, mimic_id_to_ori3, mimic_results)
    
    return slake_f_accs, slake_f_accs3


if __name__ == "__main__":
    models = ['original', 'DoLa', 'PAI', 'm3id', 'VCD', 'damro']
    # models = ["original", 'DoLa', 'PAI']
    for model in models:
        infer_path_type2 = f"{root_path}/MedHEval/type2/baselines_llava_v1.6_13b/mimic_cxr_closed/pred_{model}.jsonl"
        infer_path_type3 = f"{root_path}/MedHEval/type3/baselines_llava_v1.6_13b/pred_{model}.jsonl"
        slake_f_accs, slake_f_accs3 = eval_mimic_single(infer_path_type2, infer_path_type3)
        
        res_path = f"{root_path}/MedHEval/type2/baselines_llava_v1.6_13b/{model}_type2&3.txt"

        with open(res_path, 'w') as file:
            file.write("type2: " + str(slake_f_accs) + '\n')
            file.write("type3: " + str(slake_f_accs3) + '\n')
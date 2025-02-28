
import numpy as np
from utils.eval_yesno import evaluate_yes_no
from utils.eval_multichoice import eval_mc
import re
import json

def eval_yes_no(results):
    answers = [{"text":line['text']} for line in results]
    if len(results)>0 and 'gt' not in results[0]:
        labels = [line['gt_ans'] for line in results]
    else:
        labels = [line['gt'] for line in results]
    return evaluate_yes_no(answers, labels)


def split_choice(choices):
    # Regular expression to match the choice labels (A:, B., C))
    pattern = r'(?<!\w)([A-Z][:.])|([A-Z]\))'

    # Find all matches of the pattern
    matches = re.finditer(pattern, choices)

    # Initialize variables to store the formatted choices
    formatted_choices = []
    previous_index = 0

    # Iterate through the matches and extract choices
    for match in matches:
        start_index = match.start()
        if previous_index < start_index:
            # Append the previous choice (before the current match)
            formatted_choices.append(choices[previous_index:start_index].strip(', '))
        # Update the previous index to the end of the current match
        previous_index = start_index

    # Append the last part
    formatted_choices.append(choices[previous_index:].strip(', '))

    return formatted_choices

def avg_acc_all(accs, lens):
#     print("dqhuiudio", accs, lens)
    acc_sum = np.zeros_like(accs[0])
    len_sum = np.zeros_like(lens[0])
    for i in range(len(accs)):
        acc_sum += accs[i] * lens[i]
        len_sum += lens[i]
    avg = np.sum(acc_sum)/np.sum(len_sum)
#     print("overall:", avgs)
    return avg


def avg_acc(accs, lens):
    acc_sum = np.zeros_like(accs[0])
    len_sum = np.zeros_like(lens[0])
    s1,s2 = 0,0
#     avgs = []
    for i in range(len(accs)):
        acc_sum += accs[i] * lens[i]
        len_sum += lens[i]
#         avgs.append(np.sum(acc_sum)/np.sum(len_sum))
    return acc_sum / len_sum

def eval_closed(ori_data, id_to_ori, inference_res):
    yn_ids, mc_ids, omission_ids = [], [], []
    yn_ids_type1, yn_ids_type2, yn_ids_type3, yn_ids_type4 = [], [], [], []
    om_ids_type1, om_ids_type3 = [], []
    for i in ori_data:
        if i['question_type'] == "binary":
            if i['omission_type'] == 1 and 'yes' in i['answer'].lower():
    #         if i['answer'].lower() == "yes" and i['answer']:
                omission_ids.append(i['qid'])
                if i['hallucination_type'] == "type_1":
                    om_ids_type1.append(i['qid'])
                elif i['hallucination_type'] == "type_3":
                    om_ids_type3.append(i['qid'])
#             else:
            yn_ids.append(i['qid'])
            if i['hallucination_type'] == "type_1":
                yn_ids_type1.append(i['qid'])
            elif i['hallucination_type'] == "type_2":
                yn_ids_type2.append(i['qid'])
            elif i['hallucination_type'] == "type_3":
                yn_ids_type3.append(i['qid'])
            elif i['hallucination_type'] == "type_4":
                yn_ids_type4.append(i['qid'])
        elif i['question_type'] == "multi-choice":
            mc_ids.append(i['qid'])
#     print(len(yn_ids), len(mc_ids), len(omission_ids))
#     print("Yes-no, hallucination type numbers:", len(yn_ids_type1), len(yn_ids_type2), len(yn_ids_type3), len(yn_ids_type4))
    
    yn_results, mc_results, omission_results = [], [], []
    yn_res_type1, yn_res_type2, yn_res_type3, yn_res_type4 = [], [], [], []
    om_res_type1, om_res_type3 = [], []
    for i in inference_res:
        if i['question_id'] in yn_ids:
            yn_results.append(i)
            if i['question_id'] in yn_ids_type1:
                yn_res_type1.append(i)
            elif i['question_id'] in yn_ids_type2:
                yn_res_type2.append(i)
            elif i['question_id'] in yn_ids_type3:
                yn_res_type3.append(i)
            elif i['question_id'] in yn_ids_type4:
                yn_res_type4.append(i)
        elif i['question_id'] in mc_ids:
            ori_i = id_to_ori[i['question_id']]
            i['choices'] = ori_i['choices']
            i['hallucination_type'] = ori_i['hallucination_type']
            mc_results.append(i)
        elif i['question_id'] in omission_ids:
            omission_results.append(i)
            if i['question_id'] in om_ids_type1:
                om_res_type1.append(i)
            elif i['question_id'] in om_ids_type3:
                om_res_type3.append(i)
    len_yn_accs = [len(yn_res_type1), len(yn_res_type2), len(yn_res_type3), len(yn_res_type4)]
    len_om_accs = [len(om_res_type1), len(om_res_type3)]
#     print(len_yn_accs, len_om_accs)
#     print(yn_res_type4)
    yn_acc_all = eval_yes_no(yn_results)
    
    yn_acc_type1, yn_acc_type2, yn_acc_type3, yn_acc_type4 = eval_yes_no(yn_res_type1), eval_yes_no(yn_res_type2), eval_yes_no(yn_res_type3), eval_yes_no(yn_res_type4)
    om_acc_type1, om_acc_type3 = eval_yes_no(om_res_type1), eval_yes_no(om_res_type3)
    yn_accs = [yn_acc_type1, yn_acc_type2, yn_acc_type3, yn_acc_type4]
    
    om_accs = [om_acc_type1, om_acc_type3]
    #mc
    mc_results_process = mc_results.copy()
    for i in mc_results_process:
        for j in split_choice(i['choices']):
            if j[0] in 'ABCDEF':
                choice = 'option_' + j[0]
                i[choice] = j
        if "gt" in i:
            i['ground_truth'] = i['gt']
        elif "gt_ans" in i:
            i['ground_truth'] = i['gt_ans']
        # process the ground truth, if only with the choice label, complete it with the corresponding choice
        if len(i['ground_truth']) == 1 and i['ground_truth'] in "ABCDEF":
            try:
                ind_ = 'ABCDEF'.index(i['ground_truth'])
                key = 'option_' + 'ABCDEF'[ind_]
                i['ground_truth'] = i[key]
            except:
                print(i)
    mc_results_process_type1, mc_results_process_type2, mc_results_process_type3, mc_results_process_type4 = [], [], [], []
    for i in mc_results_process:
        if i['hallucination_type'] == "type_1":
            mc_results_process_type1.append(i)
        elif i['hallucination_type'] == "type_2":
            mc_results_process_type2.append(i)
        elif i['hallucination_type'] == "type_3":
            mc_results_process_type3.append(i)
        elif i['hallucination_type'] == "type_4":
            mc_results_process_type4.append(i)
    mc_accs = []
    len_mc_accs = []
    a=1
    for i in [mc_results_process_type1, mc_results_process_type2, mc_results_process_type3, mc_results_process_type4]:
        mc_acc = eval_mc(i, out_file=f'test_type{a}.csv')
        a+=1
        mc_accs.append(mc_acc)
        len_mc_accs.append(len(i))
    
    avg_acc = (np.array(yn_accs) * np.array(len_yn_accs) + np.array(mc_accs) * np.array(len_mc_accs)) / (np.array(len_yn_accs) + np.array(len_mc_accs))

    return avg_acc, np.array(len_yn_accs) + np.array(len_mc_accs), np.array(om_accs), np.array(len_om_accs)


def eval_all(slake_infer_path, rad_infer_path, xray_infer_path, mimic_infer_path, slake_ori, slake_id_to_ori, rad_ori, rad_id_to_ori, xray_ori, xray_id_to_ori, mimic_ori, mimic_id_to_ori):
    with open(slake_infer_path, 'r') as f:
        slake_results = [json.loads(line) for line in f]
    with open(rad_infer_path, 'r') as f:
        rad_results = [json.loads(line) for line in f]
    
    slake_f_accs, slake_f_lens, slake_o_accs, slake_o_lens = eval_closed(slake_ori, slake_id_to_ori, slake_results)
    rad_f_accs, rad_f_lens, rad_o_accs, rad_o_lens = eval_closed(rad_ori, rad_id_to_ori, rad_results)
    
    
    col1 = avg_acc([slake_f_accs, rad_f_accs], [slake_f_lens, rad_f_lens])
    col1_o = avg_acc([slake_o_accs, rad_o_accs], [slake_o_lens, rad_o_lens])
    
    with open(xray_infer_path, 'r') as f:
        xray_results = [json.loads(line) for line in f]
    with open(mimic_infer_path, 'r') as f:
        mimic_results = [json.loads(line) for line in f]
        
    mimic_f_accs, mimic_f_lens, mimic_o_accs, mimic_o_lens = eval_closed(mimic_ori, mimic_id_to_ori, mimic_results)
    xray_f_accs, xray_f_lens, xray_o_accs, xray_o_lens = eval_closed(xray_ori, xray_id_to_ori, xray_results)
    
    col2 = avg_acc([xray_f_accs, mimic_f_accs], [xray_f_lens, mimic_f_lens])
    col2_o = avg_acc([xray_o_accs, mimic_o_accs], [xray_o_lens, mimic_o_lens])
    
    col21 = avg_acc_all([xray_f_accs, mimic_f_accs], [xray_f_lens, mimic_f_lens])
    col11 = avg_acc_all([slake_f_accs, rad_f_accs], [slake_f_lens, rad_f_lens])
    
    
    return {
        "mixed": [col1, col1_o],
        "xray": [col2, col2_o],
        "avg": [col11, col21]
    }
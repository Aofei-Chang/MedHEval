from nexus.llm import BedrockClient, BedrockSettings # internal package. Not open source. 
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
import json
from langchain_core.messages import HumanMessage, SystemMessage

APP_CLIENT_ID="<<APP_CLIENT_ID>>"
settings = BedrockSettings(
    app_client_id=APP_CLIENT_ID,
    # required app_client_secret taken from environment
    app_client_secret = "app_client_secret",
    openai_azure_api_version="2023-01-01",
    api_key = "<api_key>"
    # specify any others as needed...
    # everything else inferred from environment, or use default values
)
bedrock = BedrockClient(settings=settings)
bedrock.model = "anthropic.claude-3-5-sonnet-20240620-v1:0"

system_msg_eval = """
You are provided with a dictionary containing `model_answers` from multiple different models and a `ground_truth` answer for a medical VQA task. The models answer a question about a medical knowledge in the image. Your objective is to assess each `model_answer` for hallucination tendencies. Please read and understand these instructions carefully, and keep them accessible for reference during the evaluation process.

**Evaluation Criteria:**

- **Hallucination Score (0-5):** Assign a score to each `model_answer` based on the degree of hallucination, where a higher score indicates more severe hallucination.
  - **Score 0:** The `model_answer` aligns perfectly with the `ground_truth`, providing an accurate and direct response to the input without any additional, unverified information.
  - **Score 1-2:** The `model_answer` is mostly accurate but includes minor details not present in the `ground_truth`. These additional details do not contradict known medical knowledge but are unnecessary.
  - **Score 3-4:** The `model_answer` contains significant deviations from the `ground_truth`, including information that is unverified or not supported by the question context.
  - **Score 5:** The `model_answer` is entirely incorrect, providing information that is not only unsupported by the `ground_truth` but also potentially misleading or harmful in a medical context.

**Guidelines:**

- **Directness:** Ensure that the `model_answer` addresses the input directly and succinctly.
- **Accuracy:** Verify that the information aligns with the `ground_truth` and established medical knowledge.
- **Relevance:** Avoid penalizing for additional information if it is accurate and contextually appropriate.

The input format, a json object:
{"input":str, "ground_truth": str, "models":{"model_1": model_1_answer, ...}}
Your output format:
{
"model_1": int, in range[0, 5],
"model_2": int, in range[0, 5],
...
}
"""



    
class Eval_ResList(BaseModel):
    GPT4o: int = Field(description="The evaluation results of GPT4o.", default=0)
    CheXagent: int = Field(description="The evaluation results of CheXagent.")
    LLaVA_med: int = Field(description="The evaluation results of LLaVA-med.")
    LLaVA_med_15: int = Field(description="The evaluation results of LLaVA-med_15.")
    LLaVA: int = Field(description="The evaluation results of LLaVA.")
    LLaVA_13b: int = Field(description="The evaluation results of LLaVA_13b.")
    LLM_CXR: int = Field(description="The evaluation results of LLM-CXR.")
    Med_flamingo: int = Field(description="The evaluation results of Med-flamingo.")
    MiniGPT4: int = Field(description="The evaluation results of MiniGPT4.")
    XrayGPT: int = Field(description="The evaluation results of XrayGPT.")
    RadFM: int = Field(description="The evaluation results of RadFM.")

eval_parser = PydanticOutputParser(pydantic_object=Eval_ResList)


# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a world class historian."),
#     ("user", "{input}"),
# ])


open_eval_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_msg_eval),
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="The question, ground_truth and models' model_response: {gr_pairs}. Ensure that you read the model answers carefully and evaluate its correctness. Your evaluation output in a json format, without extra explanation:",
            input_variables=["gr_pairs"],
            partial_variables={"format_instructions": eval_parser.get_format_instructions()},
        )
    ),
])


from operator import itemgetter, attrgetter

open_eval_chain = open_eval_prompt | bedrock | eval_parser



def load_model_ans(model_name):
    ans_path = f"/Users/lena/Desktop/project/intern/Hallucination_type2_open/{model_name}/open2_res.jsonl"
    with open(ans_path, "r") as f:
        results = [json.loads(line) for line in f]
    formatted_results = []
    id_results_dict = dict()
    for i in results:
        if i["question_type"] == "type4_Knowledge":
            _id = i["question_id"]
            new_dict = dict()
            # print(i)
            new_dict["ground_truth"] = i["ground_truth"]
            new_dict["question"] = i["prompt"]
            new_dict["model_answer"] = i["model_answer"]
            id_results_dict[_id] = new_dict
            formatted_results.append(new_dict)
    return formatted_results, id_results_dict


# formatted_results[:5]
from tqdm import tqdm

model_name_res_dict = dict()
model_names = ["LLM-CXR", "GPT4o", "llava-v1.6", "llava-v1.6-13b", "RadFM", "XrayGPT", "MiniGPT4", "Med-flamingo", "LLaVA-med-1.5", "LLaVA-med", "CheXagent"]
len(model_names)
for model_name in model_names:
    model_name_res_dict[model_name] = load_model_ans(model_name)[1]

final_evals = []



len(model_name_res_dict["LLM-CXR"])
question_ids = list(model_name_res_dict["LLM-CXR"].keys())
# question_ids
# for _id in tqdm(question_ids[::]):
#     gt = model_name_res_dict["LLM-CXR"][_id]["ground_truth"]
#     q = model_name_res_dict["LLM-CXR"][_id]["question"]
#     # try:
#     models_res = {
#         "CheXagent":model_name_res_dict["CheXagent"][_id]["model_answer"], 
#         "LLaVA_med":model_name_res_dict["LLaVA-med"][_id]["model_answer"], 
#         "LLaVA_med_15":model_name_res_dict["LLaVA-med-1.5"][_id]["model_answer"], 
#         "LLaVA":model_name_res_dict["llava-v1.6"][_id]["model_answer"], 
#         "LLaVA_13b":model_name_res_dict["llava-v1.6-13b"][_id]["model_answer"], 
#         "LLM_CXR":model_name_res_dict["LLM-CXR"][_id]["model_answer"], 
#         "Med_flamingo":model_name_res_dict["Med-flamingo"][_id]["model_answer"], 
#         "MiniGPT4":model_name_res_dict["MiniGPT4"][_id]["model_answer"], 
#         "XrayGPT":model_name_res_dict["XrayGPT"][_id]["model_answer"], 
#         "RadFM":model_name_res_dict["RadFM"][_id]["model_answer"]
#     }
#     # except KeyError as e:
#     #     print(e)
#     #     continue
#     if model_name_res_dict["GPT4o"].__contains__(_id):
#         models_res["GPT4o"] = model_name_res_dict["GPT4o"][_id]["model_answer"]
#     for_eval = {"input":q, "ground_truth":gt, "models": models_res}
#     # print(for_eval)
#     example_ans = open_eval_chain.invoke(dict(gr_pairs=for_eval))

#     final_evals.append(example_ans)


import concurrent
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import threading
from collections import OrderedDict

def process_single_id(args):
    _id, model_name_res_dict, open_eval_chain = args
    try:
        gt = model_name_res_dict["LLM-CXR"][_id]["ground_truth"]
        q = model_name_res_dict["LLM-CXR"][_id]["question"]
        
        models_res = {
            "CheXagent": model_name_res_dict["CheXagent"][_id]["model_answer"],
            "LLaVA_med": model_name_res_dict["LLaVA-med"][_id]["model_answer"],
            "LLaVA_med_15": model_name_res_dict["LLaVA-med-1.5"][_id]["model_answer"],
            "LLaVA": model_name_res_dict["llava-v1.6"][_id]["model_answer"],
            "LLaVA_13b": model_name_res_dict["llava-v1.6-13b"][_id]["model_answer"],
            "LLM_CXR": model_name_res_dict["LLM-CXR"][_id]["model_answer"],
            "Med_flamingo": model_name_res_dict["Med-flamingo"][_id]["model_answer"],
            "MiniGPT4": model_name_res_dict["MiniGPT4"][_id]["model_answer"],
            "XrayGPT": model_name_res_dict["XrayGPT"][_id]["model_answer"],
            "RadFM": model_name_res_dict["RadFM"][_id]["model_answer"]
        }
        
        if model_name_res_dict["GPT4o"].__contains__(_id):
            models_res["GPT4o"] = model_name_res_dict["GPT4o"][_id]["model_answer"]
            
        for_eval = {"input": q, "ground_truth": gt, "models": models_res}
        example_ans = open_eval_chain.invoke(dict(gr_pairs=for_eval))
        
        return _id, example_ans
    except Exception as e:
        print(f"Error processing ID {_id}: {str(e)}")
        return _id, None

def process_with_threads(question_ids, model_name_res_dict, open_eval_chain, num_threads=15):
    # Create an ordered dictionary to store results
    final_evals = OrderedDict()
    
    # Create arguments for each task
    args_list = [(qid, model_name_res_dict, open_eval_chain) for qid in question_ids]
    
    # Process using thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks and get futures
        future_to_id = {executor.submit(process_single_id, args): args[0] 
                       for args in args_list}
        
        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_id), 
                         total=len(question_ids)):
            _id, result = future.result()
            if result is not None:
                final_evals[_id] = result
    
    # Convert ordered dictionary to list maintaining original order
    final_results = []
    for qid in question_ids:
        if qid in final_evals:
            final_results.append(final_evals[qid])
    
    return final_results, final_evals
final_results,final_evals  = process_with_threads(question_ids, model_name_res_dict, open_eval_chain)
# Usage
import pickle
tmp_filename = f"res1.pkl" # You can choose any file name with '.pkl' extension
filehandler = open(tmp_filename, 'wb')
pickle.dump(final_evals, filehandler)
filehandler.close()

tmp_filename = f"res2.pkl" # You can choose any file name with '.pkl' extension
filehandler = open(tmp_filename, 'wb')
pickle.dump(final_results, filehandler)
filehandler.close()

    




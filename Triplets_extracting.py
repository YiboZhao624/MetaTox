'''
Totally work flow:
1. load dataset and LLMs
2. Using LLMs to extract the triplets and save them to jsonl
3. using rule based method to extract triplets and resend to LLM to filter.
4. save the filtered triplets to jsonl.
'''

'''
Parameter required:
1. dataset_folder: the path of the folder containing the dataset.
2. dataset_name: the name of the dataset. constrained in ["HateXplain", "ToxicSpans","IHC"]
3. llm_name: the name of the LLM. advice using path to the model folder.
4. output_folder: the path of the folder to save the processed data.
    **IMPORTANT**: the folder will be used as the path for the next steps.
    If you are using the step_by_step mode, please make sure the folder is correct.
    More details see the README.md
5. step_by_step: whether to use the step by step mode. default is False.
6. resume_step: the step to resume from. default is 0.
7. resume_inference: the inference to resume from. default is 0.
8. prompt_version: the version of the prompts. default is v1.
'''


import os
import re
import math
import torch
import argparse
from tqdm import tqdm
from promptsv3 import REASONPROMPTV3, TRIPLETPROMPTV3, FILTERPROMPTV3
from promptsv2 import REASONPROMPTV2, TRIPLETPROMPTV2, FILTERPROMPTV2
from prompts import REASONPROMPT, TRIPLETPROMPT, FILTERPROMPT
from utils import save_jsonl, dataset_preprocess, load_jsonl, load_llm, load_bert, setup_logger

logger = setup_logger("Extract")

def _batch_inference(llm, tokenizer, batch_data, prompt, key,ending):
    raw_inputs = []
    for ex in batch_data:
        if key:
            if isinstance(key, str):
                to_format = ex[key].strip("<|endoftext|>")
                raw_inputs.append(prompt.format(context = to_format))
            else:
                raw_inputs.append(prompt.format(
                    context=ex['text'].strip("<|endoftext|>"),
                    analysis=ex['reason'].strip("<|endoftext|>")
                ))
        else:
            raw_inputs.append(prompt.format(context = ex.strip("<|endoftext|>")))
    inputs = tokenizer(raw_inputs, return_tensors = "pt",
                       padding = True, truncation = True).to(llm.device)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens = 256)
    return tokenizer.batch_decode(outputs)

def _rulebase_triplet_extract(text):
    # Use a regular expression to find all occurrences of text within parentheses
    pattern = r'(\(.*?\))'
    matches = re.findall(pattern, text)
    res = "[" + ",".join(list(set(matches))) + "]"
    return res

def preprocess(dataset_folder, dataset_names, output_folder):
    '''
    Step 0: load dataset and save to jsonl

    saving at the output_folder with the name format of dataset_name.jsonl
    '''
    for dataset_name in dataset_names:
        data, test_data, balanced_test_data = dataset_preprocess(dataset_folder, dataset_name)
        logger.info(f"Sample data before saving: {test_data[0]}")
        save_jsonl(data, os.path.join(output_folder, f"{dataset_name}.jsonl"))
        save_jsonl(test_data, os.path.join(output_folder, f"test_{dataset_name}.jsonl"))
        save_jsonl(balanced_test_data, os.path.join(output_folder, f"balanced_test_{dataset_name}.jsonl"))
        logger.info(f"Step 0: Dataset {dataset_name} preprocessed and saved to {output_folder}")
    logger.info(f"Step 0: All datasets: {' '.join(dataset_names)} preprocessed and saved to {output_folder} with name format of dataset_name.jsonl, and test_dataset_name.jsonl and balanced_test_dataset_name.jsonl for evaluation.")

def reasoning(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version):
    '''
    Step 1: reasoning using LLM

    loading from the output_folder with the name format of dataset_name.jsonl

    saving at the output_folder with the name format of rationale_dataset_name.jsonl
    '''
    # load the llm and tokenizer first.
    llm, tokenizer = load_llm(llm_name, device)
    llm.eval()
    for dataset_name in dataset_names:
        # load the dataset.
        data = load_jsonl(os.path.join(output_folder, f"{dataset_name}.jsonl"))
        length = len(data)
        logger.info(f"Step 1: Dataset {dataset_name} loaded, total {length} samples.")

        iterate_times = math.ceil(length/batch_size)
        with torch.inference_mode():    
            for i in tqdm(range(resume_inference, iterate_times)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, length)
                
                if end_idx - start_idx < batch_size and end_idx - start_idx > 0:
                    batch_data = data[start_idx:end_idx]
                    logger.info(f"Processing last batch with {len(batch_data)} samples")
                else:
                    batch_data = data[start_idx:end_idx]
            
                if not batch_data:
                    continue
                # batch inference.

                match prompt_version:
                    case "v1":
                        prompt = REASONPROMPT
                    case "v2":
                        prompt = REASONPROMPTV2
                    case "v3":
                        prompt = REASONPROMPTV3
                    case _:
                        raise NotImplementedError
                texts = _batch_inference(llm, tokenizer, batch_data, prompt, "text", "\nAnalysis: ")

                # save the results each batch.
                for text, ex in zip(texts, batch_data):
                    ex["reason"] = text.split("<ASSISTANT>: Analysis:")[1].split("\nContext")[0].strip("<|endoftext|>")
                save_jsonl(batch_data, os.path.join(output_folder, f"rationale_{dataset_name}.jsonl"))
        # done.
        logger.info(f"Step 1: Dataset {dataset_name} processed, total {iterate_times} batches. saving at {output_folder}/rationale_{dataset_name}.jsonl")

def triplet_extracting(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version):
    '''
    Step 2: extract triplets using LLM

    loading from the output_folder with the name format of rationale_dataset_name.jsonl

    saving at the output_folder with the name format of triplets_dataset_name.jsonl
    '''
    # load the llm and tokenizer first.
    llm, tokenizer = load_llm(llm_name, device)
    llm.eval()
    for dataset_name in dataset_names:
        # load the dataset.
        data = load_jsonl(os.path.join(output_folder, f"rationale_{dataset_name}.jsonl"))
        length = len(data)
        logger.info(f"Step 2: Dataset {dataset_name} loaded, total {length} samples.")
        iterate_times = math.ceil(length/batch_size)
        with torch.inference_mode():
            for i in tqdm(range(resume_inference, iterate_times)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, length)
            
                if end_idx - start_idx < batch_size and end_idx - start_idx > 0:
                    batch_data = data[start_idx:end_idx]
                    logger.info(f"Processing last batch with {len(batch_data)} samples")
                else:
                    batch_data = data[start_idx:end_idx]
            
                if not batch_data:
                    continue
                # batch inference.
                inference_prompt = ""
                match prompt_version:
                    case "v1":
                        inference_prompt = TRIPLETPROMPT
                        key = "reason"
                    case "v2":
                        inference_prompt = TRIPLETPROMPTV2
                        key = "reason"
                    case "v3":
                        inference_prompt = TRIPLETPROMPTV3
                        key = ["text", "reason"]
                    case _:
                        raise NotImplementedError
                    
                texts = _batch_inference(llm, tokenizer, batch_data, inference_prompt, key, "\nOutput: ")
                # using such a complex ending to avoid the model repeat and generate more examples deviate the original inputs.
                for text, ex in zip(texts, batch_data):
                    ex["triplets"] = text.split("<ASSISTANT>: Output:")[1].strip("<|endoftext|>").split("\n<USER>")[0]
                    ex["triplets"] = _rulebase_triplet_extract(ex["triplets"])
                save_jsonl(batch_data, os.path.join(output_folder, f"triplets_{dataset_name}.jsonl"))
        # done.
        logger.info(f"Step 2: Dataset {dataset_name} processed, total {iterate_times} batches. saving at {output_folder}/triplets_{dataset_name}.jsonl")

def filtering(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version):
    '''
    Step 3: filter the triplets using LLM

    loading from the output_folder with the name format of triplets_dataset_name.jsonl

    saving at the output_folder with the name format of filtered_triplets_dataset_name.jsonl
    '''
    # load the llm and tokenizer first.
    llm, tokenizer = load_llm(llm_name, device)
    llm.eval()
    for dataset_name in dataset_names:
        # load the dataset.
        data = load_jsonl(os.path.join(output_folder, f"triplets_{dataset_name}.jsonl"))
        length = len(data)
        logger.info(f"Step 3: Dataset {dataset_name} loaded, total {length} samples.")
        iterate_times = math.ceil(length/batch_size)
        with torch.inference_mode():
            for i in tqdm(range(resume_inference, iterate_times)):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, length)
                
                if end_idx - start_idx < batch_size and end_idx - start_idx > 0:
                    batch_data = data[start_idx:end_idx]
                    logger.info(f"Processing last batch with {len(batch_data)} samples")
                else:
                    batch_data = data[start_idx:end_idx]
                
                if not batch_data:
                    continue
                # batch inference.
                inference_prompt = ""
                match prompt_version:
                    case "v1":
                        inference_prompt = FILTERPROMPT
                    case "v2":
                        inference_prompt = FILTERPROMPTV2
                    case "v3":
                        inference_prompt = FILTERPROMPTV3
                    case _:
                        raise NotImplementedError

                texts = _batch_inference(llm, tokenizer, batch_data, inference_prompt, "triplets", "\nOutput: ")
                for text, ex in zip(texts, batch_data):
                    ex["filtered"] = text.split("<ASSISTANT>: Output:")[1].strip("<|endoftext|>").split("\n<USER>")[0]
                    ex["filtered"] = _rulebase_triplet_extract(ex["filtered"])
                save_jsonl(batch_data, os.path.join(output_folder, f"filtered_triplets_{dataset_name}.jsonl"))
        # done.
        logger.info(f"Step 3: Dataset {dataset_name} processed, total {iterate_times} batches. saving at {output_folder}/filtered_triplets_{dataset_name}.jsonl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--dataset_names", type=str, nargs="+", required=True)
    parser.add_argument("--llm_name", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--step_by_step", type=str, default="True")

    parser.add_argument("--resume_step", type=int, default=0,
                        help="the step to do. once a step.\
        0 for the beginning. Then preprocess.\
        1 is loading the dataset from the processed data. Then analyze the reason of hate.\
        2 is loading the analyzed dataset from the processed data. Then extract triplets.\
        3 is loading the triplet file from the processed data in output folder. Then filter.")
    
    parser.add_argument("--resume_inference", type=int, default=0,
        help = "To avoid the crash during the inference, we use this parameter to resume from the last processed sample. 0 means from the beginning. U can change it according to your log and batchsize. Usually, the last tqdm number * batchsize is a good choice.")
    
    parser.add_argument("--prompt_version", type=str, default="v1",
        help = "The version of the prompts.")
    parser.add_argument("--device",type=str,default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    dataset_names = args.dataset_names
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    step_by_step = args.step_by_step
    llm_name = args.llm_name
    step_by_step = step_by_step.lower() == "true"
    resume_step = args.resume_step
    resume_inference = args.resume_inference
    device = args.device
    batch_size = args.batch_size
    prompt_version = args.prompt_version
    if not step_by_step:
        logger.warning(f"Step by step mode is False. It may takes a long time to run all steps. Please consider using step by step mode.")
        preprocess(dataset_folder, dataset_names, output_folder)
        reasoning(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
        triplet_extracting(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
        filtering(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
    else:
        match resume_step:
            case 0:
                preprocess(dataset_folder, dataset_names, output_folder)
            case 1:
                reasoning(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
            case 2:
                triplet_extracting(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
            case 3:
                filtering(dataset_names, output_folder, llm_name, device, batch_size, resume_inference, prompt_version)
            case _:
                raise ValueError(f"Step {resume_step} not implemented")



if __name__ == "__main__":
    main()
import os
import ast
import json
import random
import logging
import jsonlines
import pandas as pd
from PIL import Image
from typing import List
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = setup_logger("utils")

def load_json(path:str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
    
def load_jsonl(path:str) -> List[dict]:
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def load_llm(model_path:str, device:str) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def load_bert(model_path:str, device:str) -> tuple:
    model = AutoModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def dataset_preprocess(dataset_folder:str, dataset_name:str) -> tuple:
    '''
    load the dataset and preprocess it.
    
    available dataset:
    - HateXplain
    - ToxicSpans
    - IHC
    '''
    match dataset_name:

        case "HateXplain":
            data_path = os.path.join(dataset_folder, "hatexplain.json")
            all_data = load_dataset("json", data_files = data_path)["train"][0]
            all_data_list = [{**value, 'id': key} for key, value in all_data.items()]
            # filtered_data = {k: v for k, v in all_data.items() if v.get('rationales')}
            split_path = os.path.join(dataset_folder, "post_id_divisions.json")
            split_id = load_json(split_path)
            train_data = []
            test_data = []
            for key, value in split_id.items():
                if key == "train" or key == "val":
                    train_data.extend([ex for ex in all_data_list if ex["id"] in value])
                else:
                    test_data.extend([ex for ex in all_data_list if ex["id"] in value])

            filtered_data_list = []
            for ex in train_data:
                new_ex = {}
                new_ex["text"] = " ".join(ex["post_tokens"])
                new_ex["source"] = "hatexplain"
                if ex.get("rationales"):
                    new_ex["rationale"] = [ex["post_tokens"][i] for i in range(len(ex["rationales"])) if ex["rationales"][i] == 1]
                    new_ex["label"] = "a"
                    filtered_data_list.append(new_ex)
            
            new_test_data = []
            for ex in test_data:
                new_ex = {}
                new_ex["text"] = " ".join(ex["post_tokens"])
                new_ex["source"] = "hatexplain"
                if ex.get("rationales"):
                    new_ex["label"] = "a"
                else:
                    new_ex["label"] = "b"
                new_test_data.append(new_ex)
            
            pos_count = len([ex for ex in new_test_data if ex["label"] == "a"])
            neg_count = len([ex for ex in new_test_data if ex["label"] == "b"])
            if pos_count < neg_count:
                balance_test_data = [ex for ex in new_test_data if ex["label"] == "a"] + [ex for ex in new_test_data if ex["label"] == "b"][:pos_count]
            else:
                balance_test_data = [ex for ex in new_test_data if ex["label"] == "a"][:neg_count] + [ex for ex in new_test_data if ex["label"] == "b"]
            return filtered_data_list, new_test_data, balance_test_data

        case "ToxicSpans":
            data_path = os.path.join(dataset_folder, "toxic_spans.csv")
            test_path = os.path.join(dataset_folder, "tsd_test.csv")
            train_data = load_dataset("csv", data_files = data_path)["train"]
            test_data = load_dataset("csv", data_files = test_path)["train"]
            toxic_data = []
            all_data = []
            #train_data, test_data = data.train_test_split(test_size=0.1, seed=42).values()

            for ex in train_data:
                new_ex = {}
                new_ex["text"] = ex["text_of_post"]
                new_ex["rationale"] = [k for k in ast.literal_eval(ex["text"]).keys()]
                new_ex["source"] = "spans"
                if ex["type"]:
                    new_ex["label"] = "a"
                    toxic_data.append(new_ex)
                else:
                    new_ex["label"] = "b"
                    all_data.append(new_ex)
            
            processed_test_data = []
            for ex in test_data:
                new_ex = {}
                new_ex["text"] = ex["text_of_post"]
                new_ex["rationale"] = [k for k in ast.literal_eval(ex["text"]).keys()]
                new_ex["source"] = "spans"
                if ex["type"]:
                    new_ex["label"] = "a"
                    processed_test_data.append(new_ex)
                else:
                    new_ex["label"] = "b"
                    processed_test_data.append(new_ex)

            pos_count = len([ex for ex in processed_test_data if ex["label"] == "a"])
            neg_count = len([ex for ex in processed_test_data if ex["label"] == "b"])
            if pos_count < neg_count:
                balance_test_data = [ex for ex in processed_test_data if ex["label"] == "a"] + [ex for ex in processed_test_data if ex["label"] == "b"][:pos_count]
            else:
                balance_test_data = [ex for ex in processed_test_data if ex["label"] == "a"][:neg_count] + [ex for ex in processed_test_data if ex["label"] == "b"]
            return toxic_data, processed_test_data, balance_test_data

        case "IHC":
            data_path = os.path.join(dataset_folder, "implicit_hate_v1_stg1_posts.tsv")
            data = pd.read_csv(data_path, sep="\t")
            data_list = []
            for _, row in data.iterrows():
                ex = {}
                ex["text"] = row["post"]
                ex["source"] = "ihc"
                ex["label"] = "b" if row["class"] == "not_hate" else "a"
                data_list.append(ex)
            logger.info(f"Loaded {len(data_list)} examples from IHC")
            train_data, test_data = train_test_split(data_list, test_size=0.1, random_state=42)
            train_data = [ex for ex in train_data if ex["label"] == "a"]
            pos_count = len([ex for ex in test_data if ex["label"] == "a"])
            neg_count = len([ex for ex in test_data if ex["label"] == "b"])
            if pos_count < neg_count:
                balance_test_data = [ex for ex in test_data if ex["label"] == "a"] + [ex for ex in test_data if ex["label"] == "b"][:pos_count]
            else:
                balance_test_data = [ex for ex in test_data if ex["label"] == "a"][:neg_count] + [ex for ex in test_data if ex["label"] == "b"]
            return train_data, test_data, balance_test_data

        case _:
            raise ValueError(f"Dataset {dataset_name} not implemented")
        
def save_json(data:dict, path:str):
    with open(path, "w") as f:
        json.dump(data, f)

def save_jsonl(data:List[dict], path:str):
    with jsonlines.open(path, "a") as writer:
        for ex in data:
            writer.write(ex)

import os
import re
import math
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from utils import load_llm, load_jsonl, save_jsonl, setup_logger, load_json, save_json
from prompts import BASELINEPROMPT, RAGBASELINEPROMPT
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import random
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.manual_seed(42)
logger = setup_logger("baseline")

def print_detailed_metrics(predictions, labels, prob_results):
    new_pred = []
    new_labels = []
    
    for i in range(len(predictions)):
        if predictions[i] == "a":
            new_pred.append(1)
        else:
            new_pred.append(0)
        if labels[i] == "a":
            new_labels.append(1)
        else:
            new_labels.append(0)

    acc = np.mean(np.array(new_pred) == np.array(new_labels)).round(4)
    f1 = f1_score(new_labels, new_pred, average="macro").round(4)
    precision = precision_score(new_labels, new_pred, average="macro").round(4)
    recall = recall_score(new_labels, new_pred, average="macro").round(4)
    # AUC = roc_auc_score(new_labels, prob_results).round(4)
    precision, recall, thresholds = precision_recall_curve(new_labels, prob_results)
    PR_AUC = auc(recall, precision).round(4)

    class_f1 = f1_score(new_labels, new_pred, average=None).round(4)
    class_precision = precision_score(new_labels, new_pred, average=None).round(4)
    class_recall = recall_score(new_labels, new_pred, average=None).round(4)
    
    logger.info(f"Overall Metrics:")
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Macro F1: {f1}")
    logger.info(f"Macro Precision: {precision}")
    logger.info(f"Macro Recall: {recall}")
    # logger.info(f"AUC: {AUC}")
    logger.info(f"PR AUC: {PR_AUC}")
    logger.info("\nPer-class Metrics:")
    for i, (f, p, r) in enumerate(zip(class_f1, class_precision, class_recall)):
        logger.info(f"Class {i}:")
        logger.info(f"  F1: {f}")
        logger.info(f"  Precision: {p}")
        logger.info(f"  Recall: {r}")


def binary_classification_LLM(model, tokenizer, dataset, batch_size):
    '''
    judge whether the context is hateful or not.
    '''
    model.eval()
    with torch.no_grad():
        pred = []
        prob_results = []
        labels = []
        for i in tqdm(range(len(dataset)), desc="Processing dataset"):
            data_piece = dataset[i]
            label = data_piece["label"]
            inputs_prompts = BASELINEPROMPT
            inputs_text = inputs_prompts.format(context=data_piece["text"])
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
            outputs = model(input_ids = inputs).logits[0,-1]
            probs = torch.nn.functional.softmax(
                torch.tensor([
                    outputs[tokenizer("a").input_ids[-1]],
                    outputs[tokenizer("b").input_ids[-1]],
                ]).float(),
                dim = 0,
            ).detach().cpu().numpy()
            prob_a, prob_b = probs[0], probs[1]
            pred_label = "a" if prob_a > prob_b else "b"
            prob_results.append(prob_a)
            pred.append(pred_label)
            labels.append(label)
        print_detailed_metrics(pred, labels, prob_results)
        # print("AUC is : ", roc_auc_score(np.array(labels), np.array(pred)))
        return pred, labels
    
def binary_classification_LLM_with_ooc(model, tokenizer, dataset, batch_size):
    '''
    judge whether the context is hateful or not.
    '''
    model.eval()
    with torch.no_grad():
        length = len(dataset)
        iterate_times = math.ceil(length / batch_size)
        pred = []
        all_labels = []
        for i in tqdm(range(iterate_times), desc="Processing dataset"):
            data = dataset[i*batch_size:min((i+1)*batch_size, length)]
            labels = [data_piece["label"] for data_piece in data]
            inputs_prompts = [BASELINEPROMPT] * len(data)
            inputs_text = [prompt.format(context=data_piece["text"]) for prompt, data_piece in zip(inputs_prompts, data)]
            inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            outputs_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_text = [output[len(inputs_text[i]):] for i, output in enumerate(outputs_text)]
            for i, text in enumerate(outputs_text):
                if text.strip()[0] == "a":
                    pred.append("a")
                elif text.strip()[0] == "b":
                    pred.append("b")
                else:
                    pred.append("ooc")
                all_labels.append(labels[i])
        print("ooc count is : ", pred.count("ooc"), " percentage: ", pred.count("ooc")/len(pred))
        print("acc is : ", np.mean(np.array(pred) == np.array(all_labels)))
        print("f1 is : ", f1_score(np.array(pred), np.array(all_labels), average="macro"))
        print("precision is : ", precision_score(np.array(pred), np.array(all_labels), average="macro"))
        print("recall is : ", recall_score(np.array(pred), np.array(all_labels), average="macro"))
            
        return pred, all_labels

def _get_RAG_database(dataset_folder, dataset_name, RAGdatabase_path, bert_model_path, model_name, device_bert):
    if not os.path.exists(RAGdatabase_path):
        logger.info(f"Building RAG database for {dataset_name} at {RAGdatabase_path}")
        # 1. load dataset
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
                dataset = []
                for ex in train_data:
                    new_ex = {}
                    new_ex["text"] = " ".join(ex["post_tokens"])
                    if ex.get("rationales"):
                        new_ex["rationale"] = [ex["post_tokens"][i] for i in range(len(ex["rationales"])) if ex["rationales"][i] == 1]
                        new_ex["label"] = "a"
                    else:
                        new_ex["label"] = "b"
                    dataset.append(new_ex)
            case "ToxicSpans":
                dataset = load_dataset("csv", data_files = os.path.join(dataset_folder, "toxic_spans.csv"))["train"]
                train_data, test_data = dataset.train_test_split(test_size=0.1, seed=42).values()
                dataset = []
                for ex in train_data:
                    new_ex = {}
                    new_ex["text"] = ex["text_of_post"]
                    new_ex["source"] = "spans"
                    if ex["type"]:
                        new_ex["label"] = "a"
                    else:
                        new_ex["label"] = "b"
                    dataset.append(new_ex)
            case "HateXplain_minized":
                dataset = load_jsonl(os.path.join("replace/with/your/path/to/minized/dataset/folder", "HateXplain_minized.jsonl"))
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
                dataset = train_data
            case _:
                raise ValueError(f"Dataset {dataset_name} is not supported.")
        # 2. load bert model and tokenizer
        bert_model = AutoModel.from_pretrained(bert_model_path).to(device_bert)
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        bert_model.eval()
        # 3. build RAG database
        texts = []
        embeddings = []
        labels = []
        for data_piece in tqdm(dataset, desc="Vectorizing dataset"):
            text = data_piece["text"]
            inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device_bert)
            outputs = bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())
            labels.append(data_piece["label"])
            texts.append(text)
        embeddings = [emb.squeeze().tolist() for emb in embeddings]
        to_save_dataset = {"text":texts, "label":labels, "embedding":embeddings}
        save_json(to_save_dataset, RAGdatabase_path)
        logger.info(f"RAG database for {dataset_name} is built and saved at {RAGdatabase_path}")
        return to_save_dataset
    else:
        logger.info(f"RAG database for {dataset_name} already exists at {RAGdatabase_path}, use it directly.")
        dataset = load_json(RAGdatabase_path)
        dataset["embedding"] = [np.array(emb) for emb in dataset["embedding"]]
        return dataset

def _retrieval(query_embedding, RAG_database, top_k):
    '''
    do retrieval with query embedding and RAG database.
    '''
    # 1. calculate the similarity between query embedding and all nodes embeddings
    # 2. get the top k most similar nodes
    # 3. return the contexts of the top k nodes 
    top_k_indices = np.argsort(np.sum((np.array(RAG_database["embedding"]) - query_embedding) ** 2, axis=1))[:top_k]
    top_k_contexts = [RAG_database["text"][i] for i in top_k_indices]
    top_k_labels = [RAG_database["label"][i] for i in top_k_indices]
    first_a = next((i for i, label in enumerate(top_k_labels) if label == "a"), None)
    first_b = next((i for i, label in enumerate(top_k_labels) if label == "b"), None)
    
    if first_a is None:
        return [{"text": top_k_contexts[0], "label": "b"}, {"text": top_k_contexts[1], "label": "b"}]
    elif first_b is None:
        return [{"text": top_k_contexts[0], "label": "a"}, {"text": top_k_contexts[1], "label": "a"}]
    else:
        if first_a < first_b:
            return [{"text": top_k_contexts[first_a], "label": "a"}, {"text": top_k_contexts[first_b], "label": "b"}]
        else:
            return [{"text": top_k_contexts[first_b], "label": "b"}, {"text": top_k_contexts[first_a], "label": "a"}]


def rag_binary_classification_LLM(model, tokenizer, dataset, dataset_folder, dataset_name, RAGdatabase_path, bert_model_path, model_name, device_bert):
    '''
    judge whether the context is hateful or not.
    1. check if there are existing RAG database, if not, build it.
    2. use RAG database to do retrieval.
    3. use the retrieved context to do classification.
    '''
    model.eval()
    RAG_database = _get_RAG_database(dataset_folder, dataset_name, RAGdatabase_path, bert_model_path, model_name, device_bert)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert_model = AutoModel.from_pretrained(bert_model_path).to(device_bert)
    bert_model.eval()
    with torch.no_grad():
        pred = []
        labels = []
        all_probs = []
        for i in tqdm(range(len(dataset)), desc="Processing dataset"):
            data_piece = dataset[i]
            label = data_piece["label"]
            query = data_piece["text"]
            bert_inputs = bert_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device_bert)
            query_embedding = bert_model(**bert_inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            # 1. do retrieval
            retrieved_contexts = _retrieval(query_embedding, RAG_database, 10)
            # 2. do classification
            inputs_prompts = RAGBASELINEPROMPT.format(context=query, RAGcontext_1=retrieved_contexts[0]["text"], RAGcontext_2=retrieved_contexts[1]["text"], RAGprediction_1=retrieved_contexts[0]["label"], RAGprediction_2=retrieved_contexts[1]["label"])
            
            inputs = tokenizer(inputs_prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
            outputs = model(input_ids = inputs).logits[0,-1]
            probs = torch.nn.functional.softmax(
                torch.tensor([
                    outputs[tokenizer("a").input_ids[-1]],
                    outputs[tokenizer("b").input_ids[-1]],
                ]).float(),
                dim = 0,
            ).detach().cpu().numpy()
            all_probs.append(probs[0])
            pred_label = "a" if probs[0] > probs[1] else "b"
            pred.append(pred_label)
            labels.append(label)
        print_detailed_metrics(pred, labels, all_probs)
        return pred, labels



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="replace/with/your/path/to/qwen2_5-14b")
    parser.add_argument("--dataset_path", type=str, default="replace/with/your/path/to/test/dataset/folder/test_HateXplain.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_path", type=str, default="replace/with/your/path/to/output/folder/baseline_HateXplain.jsonl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--which_baseline", type=str, default="baseline",help="baseline or rag_baseline")
    parser.add_argument("--train_dataset_folder", type=str, default="replace/with/your/path/to/train/dataset/folder")
    parser.add_argument("--train_dataset_name", type=str, default="IHC")
    parser.add_argument("--RAGdatabase_path", type=str, default="replace/with/your/path/to/RAGdatabase/folder/RAGdatabase_IHC.jsonl")
    parser.add_argument("--bert_model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--device_bert", type=str, default="cpu")
    args = parser.parse_args()
    
    model, tokenizer = load_llm(args.model_name, args.device)
    dataset = load_jsonl(args.dataset_path)
    if args.which_baseline == "baseline":
        pred, labels = binary_classification_LLM(model, tokenizer, dataset, args.batch_size)
    elif args.which_baseline == "ooc_baseline":
        pred, labels = binary_classification_LLM_with_ooc(model, tokenizer, dataset, args.batch_size)
    elif args.which_baseline == "rag_baseline":
        pred, labels = rag_binary_classification_LLM(model, tokenizer, dataset, args.train_dataset_folder, args.train_dataset_name, args.RAGdatabase_path, args.bert_model_path, args.model_name, args.device_bert)
    save_jsonl([{"prediction":pred,"label":label} for pred, label in zip(pred, labels)], args.output_path)
    
if __name__ == "__main__":
    main()
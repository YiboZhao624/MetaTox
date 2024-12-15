import os
import json
import random
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from utils import setup_logger, load_llm, load_jsonl, save_jsonl, load_bert, load_json, save_json
from query import set_faiss_index, query_faiss_index, mapping_indices_to_nodes, NER_LLM_2, NER_LLM_4, build_graph, subgraph_search, triplets_rerank

logger = setup_logger("sample")


def dataset_sample(dataset_folder: str, dataset_name: str, sample_size: int):
    dataset_path = os.path.join(dataset_folder, "test_" + dataset_name + ".jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {dataset_folder}")
    
    dataset = load_jsonl(dataset_path)
    dataset = [data for data in dataset if data["label"] == "a"]
    random.seed(42)
    dataset = random.sample(dataset, sample_size)
    return dataset

def generate_reason_baseline(data, tokenizer, model):
    baseline_prompt = """
<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive a context. Your task is to judge whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should output a or b and then explain your reasoning.

CONTEXT: {context}

a: {{Hateful}}
b: {{Non-hatefule}}

<ASSISTANT>:Prediction: """
    input_ids = tokenizer.encode(baseline_prompt.format(context=data["text"]), return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True).split("<ASSISTANT>:")[1].strip()

def _get_RAG_database(dataset_folder, dataset_name, RAGdatabase_path, bert_model, bert_tokenizer):
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
                dataset = load_jsonl(os.path.join(dataset_folder, "HateXplain_minized.jsonl"))
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
        bert_model.eval()
        # 3. build RAG database
        texts = []
        embeddings = []
        labels = []
        for data_piece in tqdm(dataset, desc="Vectorizing dataset"):
            text = data_piece["text"]
            inputs = bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
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


def rag_binary_classification_LLM(model, tokenizer, dataset, dataset_folder, dataset_name, RAGdatabase_path, bert_model, bert_tokenizer):
    '''
    judge whether the context is hateful or not.
    1. check if there are existing RAG database, if not, build it.
    2. use RAG database to do retrieval.
    3. use the retrieved context to do classification.
    '''
    rag_reason_prompt = '''
<Instructions>: Output a or b first and then give a concise explanation for why it is hateful or not.

<USER>: You are a helpful assistant skilled in inductive reasoning and summarization. For each message, you will receive a context. Your task is to judge whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should output a or b first, then give a concise explanation for why it is hateful or not.

a: {{Hateful}}
b: {{Non-hatefule}}

Here are 2 examples on prediction:
Example 1:
<USER>: Context 1: {RAGcontext_1}
<ASSISTANT>: {RAGprediction_1}

Example 2:
<USER>: Context 2: {RAGcontext_2}
<ASSISTANT>: {RAGprediction_2}

New Input:
<USER>: Context: {context}
<ASSISTANT>:Prediction: """
'''
    model.eval()
    RAG_database = _get_RAG_database(dataset_folder, dataset_name, RAGdatabase_path, bert_model, bert_tokenizer)
    bert_model.eval()
    with torch.no_grad():
        
        for i in tqdm(range(len(dataset)), desc="Processing dataset"):
            data_piece = dataset[i]
            label = data_piece["label"]
            query = data_piece["text"]
            bert_inputs = bert_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
            query_embedding = bert_model(**bert_inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy()
            # 1. do retrieval
            retrieved_contexts = _retrieval(query_embedding, RAG_database, 10)
            # 2. do classification
            inputs_prompts = rag_reason_prompt.format(context=query, RAGcontext_1=retrieved_contexts[0]["text"], RAGcontext_2=retrieved_contexts[1]["text"], RAGprediction_1=retrieved_contexts[0]["label"], RAGprediction_2=retrieved_contexts[1]["label"])
            
            inputs = tokenizer(inputs_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            output = model.generate(**inputs, max_new_tokens=256)
            dataset[i]["reason"] = tokenizer.decode(output[0], skip_special_tokens=True,)#.split("<ASSISTANT>:")[1].strip()

        return dataset


def generate_reason_metatoxic(data, tokenizer, model, nodes, graph, index, bert, bert_tokenizer, args):
    
    match args.which_NER_prompt:
        case "NER2":
            entities = NER_LLM_2(data["text"], model, tokenizer)
        case "NER4":
            entities = NER_LLM_4(data["text"], model, tokenizer)

    bert_inputs = bert_tokenizer(entities, return_tensors="pt", padding=True, truncation=True).to(bert.device)
    bert_outputs = bert(**bert_inputs)
    entity_embedding = bert_outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()
    indices = query_faiss_index(index, entity_embedding, args.top_k, args.threshold)
    related_nodes = []
    for indice in indices:
        related_nodes.extend(mapping_indices_to_nodes(indice, nodes))
    related_nodes = list(set(related_nodes))
    related_triplets = subgraph_search(graph, related_nodes)
    query_embedding = bert_tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True).to(bert.device)
    query_embedding = bert(**query_embedding).last_hidden_state.mean(dim=1).cpu().detach().numpy()
    triplets_list_str = [f"({src}, {dst}, {rel})" for src, dst, rel in related_triplets]
    if len(triplets_list_str) > 0:  
        triplets_embeddings = bert_tokenizer(triplets_list_str, return_tensors="pt", padding=True, truncation=True).to(bert.device)
        triplets_embeddings = bert(**triplets_embeddings).last_hidden_state.mean(dim=1).cpu().detach().numpy()
        filtered_triplets_idx = triplets_rerank(query_embedding, triplets_embeddings, args.threshold)
        minimized_subgraph_str = [triplets_list_str[i] for i in filtered_triplets_idx]
        # minimized_subgraph_str = [f"({src}, {dst}, {rel})" for src, dst, rel in minimized_subgraph]
    else:
        minimized_subgraph_str = "no related triplets"
    metatoxic_prompt = """
<Instructions>: You are a helpful assistant skilled in inductive reasoning. For each message, you will receive a context and related triplets as reference. All the triplets will lead to hate. If the context contains the same meaning with one of the triplets, it is hateful. Otherwise, it is more possible to be benign but you still should leverage your knowledge to judge it. Your task is to determine whether the context is hateful or not with a single aplphabet letter among given answer choices a and b. You should first output a or b, then give a concise explanation for why it is hateful or not.

a: {{Hateful}}
b: {{Non-hatefule}}

Here are 2 examples:
Example 1:
<USER>: TRIPLETS: (niggers, is described as, stupid), 
CONTEXT: Niggers are stupid.
<ASSISTANT>: a

Example 2:
<USER>: TRIPLETS: (white woman, shows, racism), (racism, aiming at, black man)
CONTEXT: That old white lady condemns any type of racism against black men.
<ASSISTANT>: b

New Input:
<USER>: Text: {text}
Related triplets: {related_triplets}
<ASSISTANT>:"""
    input_ids = tokenizer.encode(metatoxic_prompt.format(text=data["text"], related_triplets=minimized_subgraph_str), return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)#.split("<ASSISTANT>:")[1].strip()

def prepare_metatoxic_input(args):
    triplets = json.load(open(args.triplets_path, "r"))
    triplets = triplets["triplets"]
    graph = build_graph(triplets)
    nodes_embeddings = json.load(open(args.nodes_path, "r"))
    nodes, embeddings = zip(*nodes_embeddings)
    embeddings = [list(embedding) for embedding in embeddings]
    index = set_faiss_index(embeddings)
    return nodes, graph, index
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,default="baseline",choices=["baseline","metatoxic","rag_baseline"])
    parser.add_argument("--triplets_path", type=str, required=False)
    parser.add_argument("--nodes_path", type=str, required=False)
    parser.add_argument("--which_NER_prompt", type=str, required=True, choices=["NER2", "NER4"])
    parser.add_argument("--bert_model_name", type=str, required=True, default="bert-base-uncased")
    parser.add_argument("--bert_device", type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True, default=2)
    parser.add_argument("--threshold", type=float, required=True, default=0.5)
    parser.add_argument("--case", type=str, default="False")
    parser.add_argument("--RAGdatabase_path", type=str, required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    if args.case != "True":
        dataset = dataset_sample(args.dataset_folder, args.dataset_name, args.sample_size)
        logger.info(f"Dataset sample: {args.sample_size} samples")
    else:
        dataset = [{"text": "retweeted elder lansing ( ): forget this white nationalist mess . i'm america first . so is our great ...", "source": "ihc", "label": "b"}]

        logger.info(f"case study dataset loaded: {len(dataset)} samples")

    model, tokenizer = load_llm(args.model_name, args.device)
    model.eval()
    logger.info(f"LLM loaded: {args.model_name}")


    match args.method:
        case "baseline":
            logger.info(f"Generating reasons using baseline method")
            for data in tqdm(dataset):
                data["reason"] = generate_reason_baseline(data, tokenizer, model)

        case "rag_baseline":
            logger.info(f"Generating reasons using rag baseline method")
            bert, bert_tokenizer = load_bert(args.bert_model_name, args.bert_device)
            bert.eval()
            logger.info(f"Loaded BERT model: {args.bert_model_name}")
            dataset = rag_binary_classification_LLM(model, tokenizer, dataset, args.dataset_folder, args.dataset_name, args.RAGdatabase_path, bert, bert_tokenizer)

        case "metatoxic":
            logger.info(f"Generating reasons using metatoxic method")
            nodes, graph, index = prepare_metatoxic_input(args)
            logger.info(f"Prepared nodes, graph and index")

            bert, bert_tokenizer = load_bert(args.bert_model_name, args.bert_device)
            bert.eval()
            logger.info(f"Loaded BERT model: {args.bert_model_name}")

            for data in tqdm(dataset):
                data["reason"] = generate_reason_metatoxic(data, tokenizer, model, nodes, graph, index, bert, bert_tokenizer, args)
    save_jsonl(dataset, os.path.join(args.output_folder, "test_" + args.dataset_name + "_" + args.method + "_reason.jsonl"))
    logger.info(f"Reasons saved to {os.path.join(args.output_folder, 'test_' + args.dataset_name + '_' + args.method + '_reason.jsonl')}")
if __name__ == "__main__":
    main()
'''
work flow:
0. load the query dataset.
1. using nx to build the graph from the triplets.
2. load the embedding of the nodes.
3. do NER to query.
4. find the most similar nodes to the query.
5. search the minimal connected subgraph from the graph.
6. prepare the data for the LLM.
'''
import os
import json
import math
import faiss
import argparse
import torch
from tqdm import tqdm
from typing import List
import numpy as np
import networkx as nx
from collections import deque
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc
from utils import load_llm, load_bert, setup_logger, load_json, load_jsonl   
from itertools import combinations
from prompts import GRAPHRAGPROMPT, GRAPHPROMPT_GPT, GRAPH2SHOTSPROMPT
from promptsv2 import GRAPH2SHOTSPROMPTV2, GRAPHRAGPROMPTV2
import random
import torch.backends.cudnn as cudnn

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = '42'

logger = setup_logger("Query")


def process_query_dataset(dataset_folder:str, dataset_name:str) -> List[List[str]]:
    '''
    process the query dataset
    '''
    match dataset_name:
        case "HateXplain":
            return load_jsonl(os.path.join(dataset_folder, "test_HateXplain.jsonl"))
        case "ToxicSpans":
            return load_jsonl(os.path.join(dataset_folder, "test_ToxicSpans.jsonl"))
        case "HateXplain_minized":
            return load_jsonl(os.path.join(dataset_folder, "test_HateXplain_minized.jsonl"))
        case "IHC":
            return load_jsonl(os.path.join(dataset_folder, "test_IHC.jsonl"))
        case "balanced_HateXplain":
            return load_jsonl(os.path.join(dataset_folder, "balanced_test_HateXplain.jsonl"))
        case "balanced_IHC":
            return load_jsonl(os.path.join(dataset_folder, "balanced_test_IHC.jsonl"))
        case _:
            raise ValueError(f"The dataset name {dataset_name} is not supported.")
    


def load_query_dataset(dataset_folder:str, dataset_name:List[str]) -> List[List[str]]:
    '''
    load the query dataset

    Parameters:
    dataset_folder (str): the folder of the query dataset
    dataset_name (List[str]): the name of the query dataset

    Returns:
    List[List[str]]: the content and label of the query dataset. position 0 is the content, position 1 is the label.
    '''
    query_dataset = []
    for name in dataset_name:
        logger.info(f"The dataset {name} is loading...")
        data = process_query_dataset(dataset_folder, name)
        logger.info(f"The dataset {name} is loaded with {len(data)} examples.")
        query_dataset.extend(data)
    dataset_names = " ".join(dataset_name)
    logger.info(f"The query dataset is loaded from {dataset_names} with {len(query_dataset)} examples.")
    return query_dataset

def set_faiss_index(embeddings:List[List[float]]) -> faiss.Index:
    '''
    set the faiss index
    '''
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    # logger.info(f"Embeddings shape: {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.omp_set_num_threads(1)
    index.add(embeddings)
    return index

def query_faiss_index(index:faiss.Index, query_embeddings:List[List[float]], k:int, threshold:float=0.3) -> List[List[int]]:
    '''
    Query the faiss index with multiple embeddings
    
    Parameters:
        index: faiss index
        query_embeddings: list of query embeddings, each embedding is a numpy array
        k: number of nearest neighbors to return
        threshold: the threshold of the similarity score, only return indices with similarity score >= threshold
    Returns:
        List[List[int]]: list of indices for each query embedding that have similarity scores >= threshold
    '''
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    faiss.normalize_L2(query_embeddings)
    
    search_k = min(k * 3, index.ntotal)  
    similarities, indices = index.search(query_embeddings, search_k)
    
    results = []
    for i in range(len(similarities)):
        
        scores = similarities[i]
        
        valid_mask = scores >= threshold
        valid_scores = scores[valid_mask]
        valid_indices = indices[i][valid_mask]
        
        if len(valid_scores) == 0:
            results.append([])
            continue
            
        exp_scores = np.exp(valid_scores)
        probs = exp_scores / exp_scores.sum()
        
        sorted_indices = np.argsort(probs)[::-1]  
        cumsum_probs = np.cumsum(probs[sorted_indices])
        cutoff_idx = np.searchsorted(cumsum_probs, 0.7) + 1  
        
        selected_indices = sorted_indices[:min(cutoff_idx, k)]
        results.append(valid_indices[selected_indices].tolist())
    
    return results


def mapping_indices_to_nodes(indices:List[int], nodes:List[str]) -> List[str]:
    '''
    mapping the indices to the nodes
    '''
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()

    valid_indices = [idx for idx in indices if 0 <= idx < len(nodes)]
    if len(valid_indices) < len(indices):
        logger.warning(f"Some indices were invalid: {set(indices) - set(valid_indices)}")
    return [nodes[index] for index in valid_indices]


def NER_LLM(text:str, model, tokenizer):
    '''
    use the LLM to extract the entities from the text
    '''
    prompt_template = "Extract the entities from the following text, only return the entities in the following format: entity1, entity2, entity3, ...\n {text} \nEntities:"
    inputs_text = prompt_template.format(text=text)
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    entities = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs_text):].strip()
    entities = [e.strip() for e in entities.split(",") if e.strip()]
    entities = list(set(entities))
    return entities

def NER_LLM_2(text:str, model, tokenizer):
    prompt_template = "Extract and list all distinct entities from the following text. Only return the entities as a comma-separated list like following template:\n Input: <sentence>\nOutput: <entity1>,<entity2>, <entity3> ... \nExample:\nText: u are the best gift for Muslim as you r a pig.\nEntities: gift,Muslim,pig\nText: {text}\nEntities:"
    inputs_text = prompt_template.format(text=text)
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    entities = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs_text):].strip().split("\nText")[0]
    
    entities = [e.strip() for e in entities.split(",") if e.strip()]
    entities = list(set(entities))
    # logger.info(f"Entities: {entities}")
    # logger.info(f"Text: {text}")
    return entities

def NER_LLM_3(text:str, model, tokenizer):
    prompt_template = "Extract and list all distinct entities from the following text. Only return the entities as a comma-separated list like following template:\nText: <sentence>\nEntities: <entity1>,<entity2>,<entity3>...\nExample:\nText: u are the best gift for Muslim as you r a pig.\nEntities: gift,Muslim,pig\nText: {text}\nEntities:"
    inputs_text = prompt_template.format(text=text)
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    entities = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs_text):].strip().split("\nText")[0]
    
    entities = [e.strip() for e in entities.split(",") if e.strip()]
    entities = list(set(entities))
    # logger.info(f"Entities: {entities}")
    # logger.info(f"Text: {text}")
    return entities

def NER_LLM_4(text:str, model, tokenizer):
    prompt_template = """Please extract entities from the following text.

Example 1:
Text: Muslims are terrorists and they should be banned from our country.
Entities: Muslims, terrorists, banned

Example 2:
Text: Women belong in the kitchen, not in the workplace.
Entities: Women, kitchen, workplace

Please analyze the following text:
Text: {text}
Entities:"""
    
    inputs_text = prompt_template.format(text=text)
    inputs = tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    entities = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(inputs_text):].strip().split("\nEntities")[0]
    
    entities = [e.strip() for e in entities.split(",") if e.strip()]
    entities = list(set(entities))
    # logger.info(f"Entities: {entities}")
    # logger.info(f"Text: {text}")
    return entities


def build_graph(triplets:List[List[str]]) -> nx.Graph:
    '''
    build the graph from the triplets
    '''
    graph = nx.Graph()
    for source, relation, target in triplets:
        graph.add_node(source)  # Add source node
        graph.add_node(target)  # Add target node
        graph.add_edge(source, target, relation=relation)  # Add edge with relation as attribute
    return graph

def subgraph_search(graph:nx.Graph, query:List[str]) -> nx.reportviews.EdgeView:
    '''
    search the minimal connected subgraph from the graph

    Parameters:
    graph (nx.Graph): The original graph.
    input_nodes (list): A list of nodes to connect.

    Returns:
    nx.Graph: A minimized connecting subgraph.
    '''
    minimized_subgraph = nx.Graph()
    valid_nodes = [node for node in query if node in graph.nodes]
    if len(valid_nodes) == 1:
        # if the query is a single node, return the node and its neighbors.
        neighbors = list(graph.neighbors(valid_nodes[0]))
        for neighbor in neighbors:
            relation = graph[valid_nodes[0]][neighbor]['relation']
            minimized_subgraph.add_edge(valid_nodes[0], neighbor, relation=relation)
        return minimized_subgraph.edges.data()

    for node1, node2 in combinations(valid_nodes, 2):
        if graph.has_node(node1) and graph.has_node(node2):
            try:
                # Get the shortest path between node1 and node2
                path = nx.shortest_path(graph, source=node1, target=node2)
                # if the path is too long, ignore it.
                if len(path) > 5:
                    continue
                # Add edges of the path to the minimized subgraph
                for i in range(len(path) - 1):
                    # Get the relation from the original graph
                    relation = graph[path[i]][path[i + 1]]['relation']
                    minimized_subgraph.add_edge(path[i], path[i + 1], relation=relation)
            except nx.NetworkXNoPath:
                # If there's no path between these nodes, continue, and raise a warning.
                continue
    minimized_subgraph.add_nodes_from(valid_nodes)
    return minimized_subgraph.edges.data()

def n_hop_search(graph: nx.Graph, nodes: List[str], n: int) -> nx.reportviews.EdgeView:
    minimized_subgraph = nx.Graph()
    valid_nodes = [node for node in nodes if node in graph.nodes]
    
    for start_node in valid_nodes:
        visited = set()
        queue = deque([(start_node, 0)])
        visited.add(start_node)
        
        while queue:
            current_node, hop = queue.popleft()
            if hop >= n:
                continue
            neighbors = graph.neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
                if minimized_subgraph.has_edge(current_node, neighbor):
                    continue
                relation = graph[current_node][neighbor]['relation']
                minimized_subgraph.add_edge(current_node, neighbor, relation=relation)
    return minimized_subgraph.edges.data()

def triplets_filter(query_emb, triplets_embs, threshold):
    '''
    filter the triplets based on the similarity score
    input:
        query_emb: the embedding of the query
        triplets_embs: the embeddings of the triplets
        threshold: the threshold of the similarity score
    output:
        filtered_triplets: the filtered index of triplets
    '''
    if len(query_emb.shape) == 3:
        query_emb = query_emb.mean(axis=1)

    elif len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
        
    if len(triplets_embs.shape) == 3:
        triplets_embs = triplets_embs.mean(axis=1)
    elif len(triplets_embs.shape) == 1:
        triplets_embs = triplets_embs.reshape(1, -1)
    distances = np.linalg.norm(triplets_embs - query_emb, axis=1)
    filtered_indices = np.where(distances <= threshold)[0]
    return filtered_indices

def triplets_rerank(query_emb, triplets_embs, threshold):
    '''
    rerank the triplets based on the similarity score
    input:
        query_emb: the embedding of the query
        triplets_embs: the embeddings of the triplets
        threshold: the threshold of the similarity score
    output:
        reranked_triplets_idx: the reranked index of triplets
    '''
    if len(query_emb.shape) == 3:
        query_emb = query_emb.mean(axis=1)

    elif len(query_emb.shape) == 1:
        query_emb = query_emb.reshape(1, -1)
        
    if len(triplets_embs.shape) == 3:
        triplets_embs = triplets_embs.mean(axis=1)
    elif len(triplets_embs.shape) == 1:
        triplets_embs = triplets_embs.reshape(1, -1)

    distances = triplets_embs.dot(query_emb.T)
    # logger.info(f"distances: {distances}")
    filtered_distances = distances[distances >= threshold]
    filtered_indices = np.where(distances >= threshold)[0]
    reranked_indices = filtered_indices[np.argsort(filtered_distances)]
    return reranked_indices


def prepare4LLM(query:str, triplets:nx.reportviews.EdgeView, which_prompt:str):
    '''
    prepare the data for the LLM
    
    Parameters:
    query (str): the query text
    triplets (nx.reportviews.EdgeView): the triplets searched from the graph.
    '''
    if isinstance(triplets, nx.reportviews.EdgeView):
        triplets_str = ", ".join([f"({src}, {dst}, {rel})" for src, dst, rel in triplets])
    else:
        triplets_str = triplets
    if not triplets_str:
        triplets_str = "No triplets found, it is more possible to be non-hateful."
    match which_prompt:
        case "graph":
            prompt = GRAPHRAGPROMPT.format(triplets=triplets_str, context=query)
        case "graph2shots":
            prompt = GRAPH2SHOTSPROMPT.format(triplets=triplets_str, context=query)
        case "graphGPT":
            prompt = GRAPHPROMPT_GPT.format(triplets=triplets_str, context=query)
        case "graphv2":
            prompt = GRAPHRAGPROMPTV2.format(triplets=triplets_str,context=query)
        case "graph2shotsv2":
            prompt = GRAPH2SHOTSPROMPTV2.format(triplets=triplets_str,context=query)
        case _:
            raise ValueError(f"The prompt {which_prompt} is not supported.")
    return prompt


def print_detailed_metrics(predictions, labels, probs):
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
    # AUC = roc_auc_score(new_labels, probs).round(4)
    precision, recall, thresholds = precision_recall_curve(new_labels, probs)
    PR_AUC = auc(recall, precision).round(5)



    class_f1 = f1_score(new_labels, new_pred, average=None).round(4)
    class_precision = precision_score(new_labels, new_pred, average=None).round(4)
    class_recall = recall_score(new_labels, new_pred, average=None).round(4)
    
    logger.info(f"Overall Metrics:")
    logger.info(f"Accuracy: {acc}")
    logger.info(f"Macro F1: {f1}")
    logger.info(f"Macro Precision: {precision}")
    logger.info(f"Macro Recall: {recall}")
    # logger.info(f"AUC of hate: {AUC}")
    logger.info(f"PR AUC: {PR_AUC}")

    logger.info("\nPer-class Metrics:")
    for i, (f, p, r) in enumerate(zip(class_f1, class_precision, class_recall)):
        logger.info(f"Class {i}:")
        logger.info(f"  F1: {f}")
        logger.info(f"  Precision: {p}")
        logger.info(f"  Recall: {r}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, required=True, help="the folder of the query dataset")
    parser.add_argument("--dataset_name", type=str, nargs="+", required=True, help="the name of the query dataset")
    parser.add_argument("--bert_model_path", type=str, required=True, help="the path of the bert model")
    parser.add_argument("--model_name", type=str, required=True, help="the name of the LLM model")
    parser.add_argument("--device_llm", type=str, default="cpu", help="the device to run the LLM")
    parser.add_argument("--device_bert", type=str, default="cpu", help="the device to run the BERT")
    parser.add_argument("--triplets", type=str, nargs="+", required=True, help="the path of the triplets file")
    parser.add_argument("--nodes_embeddings", type=str, required=True, nargs="+", help="the path of the nodes embeddings file")
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size")
    parser.add_argument("--which_prompt", type=str, default="graph", help="the prompt to use")
    parser.add_argument("--end_with", type=str, default="", help="the suffix of the output file")
    parser.add_argument("--which_NER_prompt", type=str, default="NER1", help="the prompt to use for NER, allow NER1, NER2, NER3, NER4.")
    parser.add_argument("--which_search", type=str, default="subgraph", help="the search method to use, allow subgraph and n_hop.")
    parser.add_argument("--n_hop", type=int, default=1, help="the number of hops to search")
    parser.add_argument("--threshold", type=float, default=0.7, help="the threshold of the similarity score, the larger, the more strict.")
    parser.add_argument("--top_k", type=int, default=2, help="the number of the most similar nodes to the query")
    parser.add_argument("--is_filter", type=str, default="False", help="whether to filter the triplets based on the similarity score")
    parser.add_argument("--is_rerank", type=str, default="False", help="whether to rerank the triplets based on the similarity score, if rerank, filter is included.")
    parser.add_argument("--rerank_threshold", type=float, default=0.5, help="the threshold of the similarity score, the larger, the more strict.")
    args = parser.parse_args()

    # step 0: load the query dataset.
    query_dataset = load_query_dataset(args.dataset_folder, args.dataset_name)
    # unzipped the query text and label.
    # query_texts, labels = zip(*query_dataset)

    triplets = []
    triplet_files = " ".join(args.triplets)
    logger.info(f"The triplets are loading from {triplet_files}...")
    for triplet_file in args.triplets:
        with open(triplet_file, "r") as f:
            data = json.load(f)
            triplets.extend(data["triplets"])
    
    # step 1: build the graph from the triplets
    graph = build_graph(triplets)
    logger.info(f"The graph is built.")

    # step 2: load the embedding of the nodes.
    nodes_embeddings = []
    nodes_embedding_files = " ".join(args.nodes_embeddings)
    logger.info(f"The nodes embeddings are loading from {nodes_embedding_files}...")
    for nodes_embedding_file in args.nodes_embeddings:
        with open(nodes_embedding_file, "r") as f:
            data = json.load(f)
            nodes_embeddings.extend(data)
    logger.info(f"The nodes embeddings are loaded.")

    # unzipped the nodes_embeddings
    nodes, embeddings = zip(*nodes_embeddings)
    embeddings = [list(embedding) for embedding in embeddings]
    index = set_faiss_index(embeddings)
    logger.info(f"The faiss index is set.")
    
    # now, we have the query texts and labels, and the nodes and embeddings. we need to execute the following steps for each query_batch.
    logger.info(f"Loading the LLM {args.model_name} at {args.device_llm}...")
    model, tokenizer = load_llm(args.model_name, args.device_llm)
    logger.info(f"The LLM is loaded.")
    logger.info(f"Loading the BERT model at {args.device_bert}...")
    bert_model, bert_tokenizer = load_bert(args.bert_model_path, args.device_bert)
    model.eval()
    bert_model.eval()
    logger.info(f"The BERT model is loaded.")

    predictions = []
    labels = []
    query_entity_count = []
    all_entitiess = []
    all_retrieved_nodes = []
    # suitable_sample = []
    all_query_results = []
    all_probs = []
    all_reasons = []
    with torch.inference_mode():
        for i, ex in tqdm(enumerate(query_dataset), total=len(query_dataset), desc="Processing the query dataset"):
            torch.cuda.empty_cache()
            query_text: str = ex["text"]
            label: str = ex["label"]
            
            # step 3: do NER to query
            match args.which_NER_prompt:
                case "NER1":
                    query_entities: List[str] = NER_LLM(query_text, model, tokenizer)
                case "NER2":
                    query_entities: List[str] = NER_LLM_2(query_text, model, tokenizer)
                case "NER3":
                    query_entities: List[str] = NER_LLM_3(query_text, model, tokenizer)
                case "NER4":
                    query_entities: List[str] = NER_LLM_4(query_text, model, tokenizer)
                case _:
                    raise ValueError(f"The NER prompt {args.which_NER_prompt} is not supported.")
            query_entity_count.append(len(query_entities))
            all_entitiess.append(query_entities)
            if len(query_entities) > 10:
                logger.warning(f"The number of entities in the query {i} is more than 10, which is {len(query_entities)}.")
            if len(query_entities) > 30:
                query_entities = query_entities[:30]
            
            if len(query_entities) == 0:
                logger.warning(f"The number of entities in the query {i} is 0, which is {len(query_entities)}.")
                prompt = prepare4LLM(query_text, "No related triplets found.", args.which_prompt)
                # go straight to LLM
                # step 8: use the LLM to predict the label
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
                outputs = model(input_ids = inputs).logits[0, -1]
                probs = torch.nn.functional.softmax(
                torch.tensor([
                        outputs[tokenizer("b").input_ids[-1]],
                        outputs[tokenizer("a").input_ids[-1]],
                    ]).float(),
                    dim=0,
                ).detach().cpu().numpy()

                # generate reason
                if i <= 1:
                    logger.info(f"prompt: {prompt}")
                    inputs_reason = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
                    outputs_reason = model.generate(**inputs_reason, max_new_tokens=100)
                    reason = tokenizer.decode(outputs_reason[0], skip_special_tokens=True)[len(prompt):].strip()
                    all_reasons.append(reason)
                all_probs.append(probs[1])
                prediction = "a" if probs[1] > probs[0] else "b"
                predictions.append(prediction)
                labels.append(label)
                # if prediction == label and len(minimized_subgraph) > 3 and len(minimized_subgraph) < 15:
                #     suitable_sample.append({"text": query_text, "label": label, "triplets": ", ".join([f"({src}, {dst}, {rel})" for src, dst, rel in minimized_subgraph])})
                if i <= 10:
                    # logger.info(f"reason: {reason}")
                    logger.info(f"prompt: {prompt}")
                    logger.info(f"probs: {probs[1]}")
                    logger.info(f"prediction: {prediction}")
                    logger.info(f"label: {label}")
                    logger.info(f"before filter: {len(all_query_results[-1])}")
                    logger.info(f"after filter: {len(minimized_subgraph)}")
                continue

            # step 4: find the most similar nodes to the query
            bert_inputs = bert_tokenizer(query_entities, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
            bert_outputs = bert_model(**bert_inputs)
            entity_embedding = bert_outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()

            # step 5: find the most similar nodes to the query
            indices = []
            if len(entity_embedding) > 0:
                # logger.info(f"entity_embedding shape: {np.array(entity_embedding).shape}")
                indices = query_faiss_index(index, entity_embedding, args.top_k, args.threshold)
            else:
                indices = []

            most_similar_nodes = []
            original_nodes = nodes 
            for indice in indices:
                mapped_nodes = mapping_indices_to_nodes(indice, original_nodes)
                most_similar_nodes.extend(mapped_nodes)

            most_similar_nodes = list(set(most_similar_nodes))
            # very important. The LLM is sensitive to the order of the nodes input.
            # without sorting, the performance will be unstable.
            most_similar_nodes.sort()  
            all_retrieved_nodes.append(most_similar_nodes)
            match args.which_search:
                case "subgraph":
                    if i == 0:
                        logger.info("subgraph search")
                    # step 6: search the minimal connected subgraph from the graph
                    minimized_subgraph = subgraph_search(graph, most_similar_nodes)
                case "n_hop":
                    if i == 0:
                        logger.info("n_hop search")
                    # step 6: search the n-hop subgraph from the graph
                    minimized_subgraph = n_hop_search(graph, most_similar_nodes, args.n_hop)
                case _:
                    raise ValueError(f"The search method {args.which_search} is not supported.")
            all_query_results.append(", ".join([f"({sr}, {ds}, {re})" for sr, ds, re in minimized_subgraph]))
            # step 6.5: filter the triplets based on the similarity score
            if args.is_filter == "True" and len(minimized_subgraph) > 0:    
                query_embedding = bert_tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
                query_embedding = bert_model(**query_embedding).last_hidden_state.mean(dim=1).cpu().detach().numpy()
                normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
                triplets_list_str = [f"({src}, {dst}, {rel})" for src, dst, rel in minimized_subgraph]
                if len(triplets_list_str) > 0:  
                    triplets_embeddings = bert_tokenizer(triplets_list_str, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
                    triplets_embeddings = bert_model(**triplets_embeddings).last_hidden_state.mean(dim=1).cpu().detach().numpy()
                    normalized_triplets_embeddings = triplets_embeddings / np.linalg.norm(triplets_embeddings, axis=1, keepdims=True)
                    filtered_triplets_idx = triplets_filter(normalized_query_embedding, normalized_triplets_embeddings, args.filter_threshold)
                    minimized_subgraph = [triplets_list_str[i] for i in filtered_triplets_idx]
                else:
                    logger.warning(f"The number of triplets is 0, so no need to filter.")
            if args.is_rerank == "True" and len(minimized_subgraph) > 0:
                query_embedding = bert_tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
                query_embedding = bert_model(**query_embedding).last_hidden_state.mean(dim=1).cpu().detach().numpy()
                normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)
                triplets_list_str = [f"({src}, {dst}, {rel})" for src, dst, rel in minimized_subgraph]
                if len(triplets_list_str) > 0:
                    triplets_embeddings = bert_tokenizer(triplets_list_str, return_tensors="pt", padding=True, truncation=True).to(bert_model.device)
                    triplets_embeddings = bert_model(**triplets_embeddings).last_hidden_state.mean(dim=1).cpu().detach().numpy()
                    normalized_triplets_embeddings = triplets_embeddings / np.linalg.norm(triplets_embeddings, axis=1, keepdims=True)
                    reranked_triplets_idx = triplets_rerank(normalized_query_embedding, normalized_triplets_embeddings, args.rerank_threshold)
                    # logger.info(f"reranked_triplets_idx: {reranked_triplets_idx}")
                    # logger.info(f"triplets_list_str: {triplets_list_str}")
                    # logger.info(f"type of reranked_triplets_idx: {type(reranked_triplets_idx)}")
                    minimized_subgraph = [triplets_list_str[i] for i in reranked_triplets_idx]
            if len(minimized_subgraph) == 0:
                logger.warning(f"No related triplets found for the query {i}.")
            else:
                logger.info(f"The number of triplets is {len(minimized_subgraph)} for the query {i}.")
            # step 7: prepare the data for the LLM
            prompt = prepare4LLM(query_text, minimized_subgraph, args.which_prompt)

            # step 8: use the LLM to predict the label
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
            outputs = model(input_ids = inputs).logits[0, -1]
            probs = torch.nn.functional.softmax(
            torch.tensor([
                    outputs[tokenizer("b").input_ids[-1]],
                    outputs[tokenizer("a").input_ids[-1]],
                ]).float(),
                dim=0,
            ).detach().cpu().numpy()

            # generate reason
            if i <= 1:
                logger.info(f"prompt: {prompt}")
                inputs_reason = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
                outputs_reason = model.generate(**inputs_reason, max_new_tokens=100)
                reason = tokenizer.decode(outputs_reason[0], skip_special_tokens=True)[len(prompt):].strip()
                all_reasons.append(reason)
            all_probs.append(probs[1])
            prediction = "a" if probs[1] > probs[0] else "b"
            predictions.append(prediction)
            labels.append(label)
            # if prediction == label and len(minimized_subgraph) > 3 and len(minimized_subgraph) < 15:
            #     suitable_sample.append({"text": query_text, "label": label, "triplets": ", ".join([f"({src}, {dst}, {rel})" for src, dst, rel in minimized_subgraph])})
            if i <= 10:
                # logger.info(f"reason: {reason}")
                logger.info(f"prompt: {prompt}")
                logger.info(f"probs: {probs[1]}")
                logger.info(f"prediction: {prediction}")
                logger.info(f"label: {label}")
                logger.info(f"before filter: {len(all_query_results[-1])}")
                logger.info(f"after filter: {len(minimized_subgraph)}")


    # with open(args.dataset_folder + f"/suitable_sample_{args.end_with}.jsonl", "w") as f:
    #     for sample in suitable_sample:
    #         f.write(json.dumps(sample) + "\n")
    # logger.info(f"The suitable samples are saved to {args.dataset_folder}/suitable_sample_{args.end_with}.jsonl")
    
    # step 10: evaluate the predictions
    try:
        with open(args.dataset_folder + f"/{args.dataset_name}_predictions_{args.end_with}.jsonl", "w") as f:
            for prediction, label, entity_count, entities, query_result, retrieved_nodes, prob in zip(predictions, labels, query_entity_count, all_entitiess, all_query_results, all_retrieved_nodes, all_probs):
                f.write(json.dumps({"prediction": prediction, "label": label, "entity_count": entity_count, "entities": entities, "query_result": query_result,"retrieved_nodes": retrieved_nodes, "prob": str(prob)}) + "\n")
    except Exception as e:
        logger.error(f"Error writing to file: {e}")
        with open(args.dataset_folder + f"/{args.dataset_name}_predictions_{args.end_with}.jsonl", "w") as f:
            for prediction, label, entity_count, entities, query_result, retrieved_nodes in zip(predictions, labels, query_entity_count, all_entitiess, all_query_results, all_retrieved_nodes):
                f.write(json.dumps({"prediction": prediction, "label": label, "entity_count": entity_count, "entities": entities, "query_result": query_result,"retrieved_nodes": retrieved_nodes}) + "\n")
    logger.info(f"The predictions are saved to {args.dataset_folder}/{args.dataset_name}_predictions_{args.end_with}.jsonl")

    print_detailed_metrics(predictions, labels, all_probs)

if __name__ == "__main__":
    main()
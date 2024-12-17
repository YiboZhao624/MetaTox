import re
import json
import torch
import logging
import argparse
import jsonlines
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
from utils import setup_logger

logger = setup_logger(__name__)

HF_TOKEN = "REPLACE_WITH_YOUR_HF_TOKEN"

def rulebase_triplet_extract(text):
    # Use a regular expression to find all occurrences of text within parentheses
    pattern = r'(\(.*?\))'
    matches = re.findall(pattern, text)
    triplets = []
    for match in matches:
        elements = match.split(',')
        if len(elements) == 3:
            triplet = tuple(elem.strip().strip(")").strip("(") for elem in elements)
            triplets.append(triplet)
    
    return triplets

def load_model(model_name, device):
    tokenizer = BertTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = BertModel.from_pretrained(model_name, token=HF_TOKEN).to(device)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model

def load_collated_data(folder_path,dataset):
    data = []
    try:
        with jsonlines.open(folder_path + f"/filtered_triplets_{dataset}.jsonl", "r") as f:
            for line in f:
                data.append(line["filtered"])
        logger.info(f"Loaded triplet data {dataset} successfully.")
    except Exception as e:
        logger.error(f"Error loading collated data: {e}")
    return data

def load_triplets(dataset):
    res = []
    for entry in dataset:
        triplets = rulebase_triplet_extract(entry)
        res.extend(triplets)
    return res

def extract_nodes(triplets):
    nodes = set()
    for triplet in triplets:
        nodes.add(triplet[0])  # subject
        nodes.add(triplet[1])  # relation
        nodes.add(triplet[2])  # object
    return list(nodes)

def get_embeddings(nodes, tokenizer, model,device):
    embeddings = []
    for node in tqdm(nodes, desc="Getting nodes' embeddings"):
        inputs = tokenizer(node, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embedding)
    return np.vstack(embeddings)

def cluster_entities(embeddings, distance_threshold=0.5):

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='ward',
        metric='euclidean'
    )
    
    cluster_labels = clustering.fit_predict(embeddings)
    
    unique_labels = np.unique(cluster_labels)
    cluster_centers = np.array([
        embeddings[cluster_labels == label].mean(axis=0)
        for label in unique_labels
    ])
    
    return cluster_labels, cluster_centers

def merge_similar_entities(nodes, embeddings, distance_threshold=0.5):
    cluster_labels, _ = cluster_entities(embeddings, distance_threshold)
    
    merged_entities = {}
    clusters = {}
    
    for i, (node, label) in enumerate(zip(nodes, cluster_labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((node, embeddings[i]))
    

    for label, cluster_nodes in clusters.items():
        representative = max(cluster_nodes, key=lambda x: len(x[0]))[0]
        for node, _ in cluster_nodes:
            merged_entities[node] = representative
            
    return merged_entities, clusters

def display_merged_examples(clusters, num_examples=5):
    logger.info(f"\n{'='*50}\nmerged examples (display {num_examples} clusters):")
    
    multi_entity_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    
    for i, (label, cluster_nodes) in enumerate(list(multi_entity_clusters.items())[:num_examples]):
        nodes = [node[0] for node in cluster_nodes]
        logger.info(f"contain entities: {nodes}")
        logger.info(f"merged entity: {max(nodes, key=len)}")
        logger.info("-" * 30)

def merge_triplets(triplets, merged_entities):
    merged_triplets = []
    for triplet in triplets:
        merged_triplet = (
            merged_entities.get(triplet[0], triplet[0]),  # subject
            merged_entities.get(triplet[1], triplet[1]),  # relation
            merged_entities.get(triplet[2], triplet[2])   # object
        )
        merged_triplets.append(merged_triplet)
    return merged_triplets

def save_merged_results(merged_triplets, output_path, dataset):
    output_data = {
        "num_triplets": len(merged_triplets),
        "triplets": merged_triplets
    }
    
    output_file = f"{output_path}/{dataset}_merged_triplets.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Merged triplets saved to {output_file}")

def merge_dataset(args):
    data_hatexplain = load_collated_data(args.folder_path,"HateXplain")
    data_toxicspans = load_collated_data(args.folder_path,"ToxicSpans")
    data_IHC = load_collated_data(args.folder_path,"IHC")
    data = data_hatexplain + data_toxicspans + data_IHC
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="The name of the model to use for embedding.")
    parser.add_argument("--folder_path", type=str,
                        default="Your/path/to/processed_data/text_speech",
                        help="The path to the folder containing the dataset.")
    parser.add_argument("--distance_threshold", type=float, default=0.5,
                        help="The distance threshold for clustering.")
    parser.add_argument("--dataset", type=str, default="toxicspans",
                        help="The name of the dataset to use. within toxicspans,hatexplain,IHC.")
    parser.add_argument("--output_path", type=str,
                        default="Your/path/to/save/merged_entities",
                        help="The path to the folder to save the merged entities.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to use for embedding.")
    parser.add_argument("--merge", type = str, default="False",help="Whether to merge the two datasets")
    args = parser.parse_args()
    
    if args.merge == "True":
        data = merge_dataset(args)
    else:
        data = load_collated_data(args.folder_path,args.dataset)

    logger.info(f"Loading model {args.model_name} at {args.device}...")
    tokenizer, model = load_model(args.model_name,args.device)
    # data preprocessing
    triplets = load_triplets(data)
    nodes = extract_nodes(triplets)
    logger.info(f"Extracted {len(nodes)} nodes from {args.dataset} triplets before merging.")
    
    # get embeddings and clustering and merging.
    embeddings = get_embeddings(nodes, tokenizer, model,args.device)
    merged_entities, clusters = merge_similar_entities(nodes, embeddings, args.distance_threshold)
    logger.info(f"Left {len(clusters.items())} entities after merging from {args.dataset} triplets.")
    display_merged_examples(clusters)
    merged_triplets = merge_triplets(triplets, merged_entities)

    logger.info(f"Merged {len(merged_triplets)} triplets from {args.dataset} triplets.")
    merged_triplets = list(set(merged_triplets))
    logger.info(f"Left {len(merged_triplets)} unique triplets after merging from {args.dataset} triplets.")

    if args.merge == "True":
        save_merged_results(merged_triplets, args.output_path, "merged_3dataset")
    else:
        save_merged_results(merged_triplets, args.output_path, args.dataset)

    # save the merged entities
    if args.merge == "True":
        with open(f"{args.output_path}/merged_3dataset_merged_entities.json", "w") as f:
            json.dump(merged_entities, f, indent=4)
        logger.info(f"Merged entities saved to {args.output_path}/merged_3dataset_merged_entities.json")
    else:
        with open(f"{args.output_path}/{args.dataset}_merged_entities.json", "w") as f:
            json.dump(merged_entities, f, indent=4)
        logger.info(f"Merged entities saved to {args.output_path}/{args.dataset}_merged_entities.json")

    res_nodes = [triplet[0] for triplet in merged_triplets]
    res_nodes.extend([triplet[2] for triplet in merged_triplets])
    res_nodes = list(set(res_nodes))
    logger.info(f"Left {len(res_nodes)} unique nodes after merging from {args.dataset} triplets.")
    node_embeddings = get_embeddings(res_nodes, tokenizer, model,args.device)
    to_save = list(zip(res_nodes, [embeddings.tolist() for embeddings in node_embeddings]))
    if args.merge == "True":
        with open(f"{args.output_path}/merged_3dataset_nodes_embeddings.json", "w") as f:
            json.dump(to_save, f, indent=4)
        logger.info(f"Merged nodes embeddings saved to {args.output_path}/merged_3dataset_nodes_embeddings.json")
    else:
        with open(f"{args.output_path}/{args.dataset}_nodes_embeddings.json", "w") as f:
            json.dump(to_save, f, indent=4)
        logger.info(f"Merged nodes embeddings saved to {args.output_path}/{args.dataset}_nodes_embeddings.json")


if __name__ == "__main__":
    main()

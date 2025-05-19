# ***MetaTox***: Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph

## 1. Introduction

This repository contains the code for the paper "Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph". The file structure is as follows:

```
MetaTox/
├── datafolder/   # An empty folder for storing the data
├── README.md 
├── requirements.txt
├── Triplets_extracting.py
├── Entity_linking.py 
├── query.py 
├── reason.py 
└── utils.py 
```

## 2. Requirements

You can install the requirements by running the following command:
```
pip install -r requirements.txt
```

## 3. Usage

### 3.1. Data Collection

Due to the used datasets are well established by the previous works, we do not provide the data collection code in this repository. To get the data, please refer to the following links:

- [HateXplain](https://github.com/hate-alert/HateXplain)
- [IHC](https://github.com/SALT-NLP/implicit-hate)
- [ToxicSpans](https://github.com/google-research/toxic-spans)

To ensure the reproducibility of our experiments, we adopted the data split as HateXplain provided ids. For the rest two datasets, we used the train-test split with random seed 42, which is written in the code.

The meta-toxic knowledge graph built by Qwen2.5-14B-Instruct can be acquired by sending an email to [Yibo Zhao](yibozhao@stu.ecnu.edu.cn).

### 3.2. Graph Construction

To construct the knowledge graph, we first need to extract the triplets from the data. The code for extracting the triplets is in the file `Triplets_extracting.py`. It is corresponding to the Section 3.2 in the paper.

The code includes 4 steps:

- Preprocessing: Preprocess the data. Mainly includes the data split and filtering the non-toxic data in the training set.
- Reasoning: Generate the reasoning results for the filtered training set.
- Triplets Extracting: Extract the triplets from the original data and the reasoning results.
- Filtering: Filter the triplets which are non-toxic.

To run the code, please use the following command:

```
python Triplets_extracting.py
    --dataset_folder /path/to/your/dataset/  \
    --dataset_names HateXplain IHC ToxicSpans \
    --output_folder /path/to/your/output/ \
    --llm_name /path/to/your/llm/ \
    --device your_device \
    --step_by_step False \
    --prompt_version v3 \
    --batch_size 8
```

This processing will take a long time. If you want to avoid any potential issues like cuda out of memory, you can set the `step_by_step` to `True` and add `resume_step` parameter. This will process the data step by step. If the step is not completed, you can resume from the last batch with the `resume_inference` parameter. You can see the detailed description in the code.

After running the code, you will get the triplets extracted from the training set. Then we need to execute the Entity_linking.py to remove duplicates by running the following command:

```
python Entity_linking.py
    --folder_path /path/to/your/output/ \
    --dataset toxicspans IHC HateXplain \
    --output_path /path/to/your/output/ \
    --device your_device \
```

### 3.3 Graph Query

To query the graph, you can use the `query.py` file. This is corresponding to the Section 3.3 in the paper. You can see the detailed description in the code. To reproduce the results of our experiments, taking HateXplain based on Qwen2_5-14B as an example, you can use the following command:

```
nohup bash query.sh &
```

This will take some time depending on your GPU. The log will be saved in the `loggings` folder.

### 3.4. Reasoning

This part is corresponding to the Section 4.3 in the paper. You can simply run the `reason.py` file to get the reasoning results.

### 3.5. Baseline

To reproduce the baseline results, you can use the `baseline.py` file. This is corresponding to the Section 4.1 in the paper. You can see the detailed description in the code.

## Citation

If this work is related or useful for your research, please cite:

```
@inproceedings{
2025enhancing,
title={Enhancing {LLM}-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph},
author={Yibo Zhao, Jiapeng Zhu, Can Xu, Yao Liu, Xiang Li},
booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
year={2025},
url={https://openreview.net/forum?id=pUpPJqDgZz}
}
```

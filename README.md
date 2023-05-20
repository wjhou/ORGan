# <span style="font-variant:small-caps;">ORGan</span>: Observation-Guided Radiology Report Generation via Tree Reasoning

This repository is the implementation of *<span style="font-variant:small-caps;">ORGan</span>: Observation-Guided Radiology Report Generation via Tree Reasoning*.
Before running the code, please install the prerequisite libraries, and follow Step 0, Step 1, and Step 2 to replicate the experiments.

## Overview
This paper explores the task of radiology report generation, which aims at generating free-text descriptions for a set of radiographs. One significant challenge of this task is how to correctly maintain the consistency between the images and the lengthy report. Previous research explored solving this issue through planning-based methods, which generate reports only based on high-level plans. However, these plans usually only contain the major observations from the radiographs (e.g., lung opacity), lacking much necessary information, such as the observation characteristics and preliminary clinical diagnoses. To address this problem, the system should also take the image information into account together with the textual plan and perform stronger reasoning during the generation process. In this paper, we propose an Observation-guided radiology Report Generation framework (**<span style="font-variant:small-caps;">ORGan</span>**). It first produces an observation plan and then feeds both the plan and radiographs for report generation, where an observation graph and a tree reasoning mechanism are adopted to precisely enrich the plan information by capturing the multi-formats of each observation. Experimental results demonstrate that our framework outperforms previous state-of-the-art methods regarding text quality and clinical efficacy.
![Alt text](figure/overview.png?raw=true "Title")

## Requirements
- `torch==1.9.1`
- `torchvision==0.10.1`
- `transformers==4.15.0`

## Step 0: Data Preparation and Observation Plan/Graph Extraction
Please download the two datasets: [IU-Xray](https://openi.nlm.nih.gov/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). For observation preprocessing, we use [CheXbert](https://arxiv.org/pdf/2004.09167.pdf) to extract relevant observation information. Please follow the [instruction](https://github.com/stanfordmlgroup/CheXbert#prerequisites) to extract the observation tags.

### Step 0.1 Observation Graph Extraction
```
chmod +x ./src/graph_construction/run_iu_xray.sh
./src/graph_construction/run_iu_xray.sh
```

### Step 0.2 Observation Plan Extraction
```
cd ORGan
python ./src/plan_extraction.py
```

## Step 1: Observation Planning
There are two parameters required to run the code of planner: 
- `debug: whether debugging the code (0 for debugging and 1 for running)`
- `checkpoint_name: indicating the location for the pre-trained visual model, mainly for IU Xray dataset`.
```
chmod +x ./script_plan/run_iu_xray.sh
./script_plan/run_iu_xray.sh debug checkpoint_name
```

## Step 2: Observation-guided Report Generation
There are four parameters required to run the code of generator:
- `debug: whether debugging the code (0 for debugging and 1 for running)`
- `plan_model_name_or_path: indicating the location of trained planner (from Step 1)`
- `plan_eval_file: indicating the file name of generated plans for the validation set (from Step 1)`
- `checkpoint_name: indicating the location for the pre-trained visual model, mainly for IU－Xray dataset, same as the setting of the planner`
```
chmod +x ./script/run_iu_xray.sh
./script/run_iu_xray.sh debug plan_model_name_or_path plan_eval_file checkpoint_name
```

## Citation
If you use the <span style="font-variant:small-caps;">ORGan</span>, please cite our paper:
```
@inproceedings{ORGan,
	title        = {ORGan: Observation-Guided Radiology Report Generation via Tree Reasoning},
	author       = {Hou, Wenjun and Xu, Kaishuai and Cheng, Yi and Li, Wenjie and Liu, Jiang}, 
	year         = 2023,
	month        = jul,
	booktitle    = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics}
}
```
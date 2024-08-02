# Duplicate-Bug-Detection

[![DOI](https://zenodo.org/badge/678604821.svg)](https://zenodo.org/badge/latestdoi/678604821)

## About Project
```
This repository contains the implementation of our paper "A Comparative Study of Text Embedding Models for Semantic Text Similarity in Bug Reports".
Duplicate bug report detection in tracking systems saves debugging efforts. Traditional solutions lack clear ranking, 
deterring their use. Our paper introduces an NLP-based method using bug report attributes, leveraging a neural 
network for retrieval.
```

### If you use any of this code please Cite Our Paper
```
@INPROCEEDINGS{10512000,
  author={Patil, Avinash and Han, Kihwan and Jadon, Aryan},
  booktitle={2024 11th International Conference on Signal Processing and Integrated Networks (SPIN)}, 
  title={A Comparative Analysis of Text Embedding Models for Bug Report Semantic Similarity}, 
  year={2024},
  volume={},
  number={},
  pages={262-267},
  keywords={Training;Analytical models;Limiting;Databases;Computer bugs;Semantics;Software systems;ADA;BERT;Bug Reports;Defect Reports;Duplicate Detection;Embeddings;FastText;Gensim;GPT3;GPT3.5;Information Retrieval;Large Language Models;LLM;Natural Language Processing;Sentence Textual Similarity;Similarity Search},
  doi={10.1109/SPIN60856.2024.10512000}}
```

## Installation Instructions

### Pre-requisites:
- Make sure you have Python and pip installed.
  
Steps:

1. **Clone the repository (if you haven't already)**:
   ```bash
   git clone https://github.com/av9ash/DuplicateBugDetection.git
   cd DuplicateBugDetection
   ```

2. **(Optional but recommended) Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

After following these steps, all necessary libraries should be installed, and you're ready to execute the project.


### Steps to Execute the Project

1. This is the first script you'll need to run. Its main purpose is to create a mapping of bugs, 
which can be utilized in subsequent scripts.
    ```bash
    python create_bugs_map.py
    ```
   
2. After generating the bugs map, the next step is to split the data into training and testing datasets.
    ```bash
    python create_train_test.py
    ```
   
3. Once you have the training and testing data ready, this is the final script you will run which might contain the main 
   algorithm or process of the project.
    ```bash
   python main_file.py
   ```

# Assignment 2 – Named Entity Recognition (NER)

> **Objective:** Build, train, and evaluate LSTM-based Named Entity Recognition models on the ReCoNLL-2003 dataset using various architectures and pretrained embeddings.

---

## Overview
This project implements a complete **Named Entity Recognition (NER)** pipeline using neural architectures built from scratch in **PyTorch**.  
It covers every step — from data preprocessing and vocabulary creation to model training, evaluation, and error analysis.

The models tag tokens in English sentences with entity types:
- **PER** – Person  
- **ORG** – Organization  
- **LOC** – Location  

---

## Key Topics Covered
- CoNLL data preprocessing  
- Vocabulary creation and OOV handling  
- LSTM and BiLSTM architectures  
- Hyperparameter tuning and evaluation  
- Integration of pretrained GloVe embeddings  
- Confusion matrices and error analysis 

---

##  Dataset
**Dataset:** [ReCoNLL-2003 (Corrected CoNLL-2003)](https://www.clips.uantwerpen.be/conll2003/ner/)  
**Files:**  
- `train.txt`, `dev.txt`, `test.txt` (downloaded automatically via `wget` commands)  
**Annotation Scheme:** `IOB` (BIO)

Each line contains a token and its tag, and sentences are separated by blank lines.

---

## Pipeline Overview

### 1. Data Preparation
- **`read_data()`** parses CoNLL-style files into `(tokens, tags)` pairs.  
- **`Vocab` class** builds word/tag dictionaries with special tokens:  
  `__pad__` (0) and `__unk__` (1).  
- **`prepare_data()`** converts words and tags to indices, applies padding to max length.  
- **`SequenceDataset`** wraps the indexed sequences for PyTorch DataLoader.

### 2. Model Architecture
Implemented via the `NERNet` class:
```python
class NERNet(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, directions):
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=n_layers,
                            bidirectional=(directions == 2), batch_first=True)
        self.fc = nn.Linear(hidden_size * directions, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
```

### 3. Training
- Optimizer: **Adam**
- Loss: **CrossEntropyLoss** (ignores padding index 7)
- Epochs: **5 (base)** and **10 (best models)**
- Metrics recorded: training & dev loss, accuracy.

### 4. Evaluation
The `evaluate()` function computes:
- **Precision**, **Recall**, and **F1-score** (weighted)
- With and without “O” tag
- **Confusion matrices** for visual inspection
- **Classification reports** via scikit-learn

### 5. Experiments
Nine models trained with varying LSTM depth, hidden size, and directionality:
| Model | Hidden Size | Layers | Directions | Embedding | Epochs | Best F1 (w/o O) |
|:------|-------------:|:-------:|:-----------:|:-----------|:-------:|:----------------|
| 1 | 500 | 1 | 1 | Learned | 5 | 0.84 |
| 2 | 500 | 2 | 1 | Learned | 5 | 0.86 |
| 3 | 500 | 3 | 1 | Learned | 5 | 0.86 |
| 4 | 500 | 1 | 2 | Learned | 5 | 0.87 |
| 5 | 500 | 2 | 2 | Learned | 5 | 0.89 |
| 6 | 500 | 3 | 2 | Learned | 5 | **0.90** |
| 7 | 800 | 1 | 2 | Learned | 5 | 0.89 |
| 8 | 800 | 2 | 2 | Learned | 5 | 0.91 |
| 9 | 800 | 3 | 2 | Learned | 5 | **0.92** |

> **Best Learned Model:** Model 9 (`hidden=800, layers=3, bidirectional=True`)

---

## Pretrained Embeddings
To boost performance, GloVe embeddings were integrated:
```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```
The script loads `glove.6B.300d.txt` and initializes the embedding layer:
```python
emb_matrix = get_emb_matrix('glove.6B.300d.txt', vocab)
initialize_from_pretrained_emb(model, emb_matrix)
```

| Model | Embedding | Hidden | Layers | Directions | F1 (w/o O) |
|:-------|:-----------|:------:|:-------:|:------------:|:------------:|
| 9 | GloVe-300d | 800 | 3 | 2 | **0.95**  |

> GloVe pretraining improved generalization and entity consistency significantly.

---

## 📊 Evaluation Metrics
**Weighted (All tags):**  
- Precision: 0.93  
- Recall: 0.92  
- F1-score: **0.92**

**Without “O”:**  
- Precision: 0.95  
- Recall: 0.94  
- F1-score: **0.95**

---

## Error Analysis
Using `simple_analyze_errors()` and `print_error_analysis()`:
- Common confusions: `LOC` ↔ `ORG`
- Occasional missed continuations of multi-token entities (`I-PER`)
- False positives on capitalized non-entities

### Suggested Improvements
1. Add **CRF layer** to improve sequence consistency.  
2. Introduce **gazetteer features** for `LOC` and `ORG`.  
3. Increase **O-tag weight** or use **focal loss** to reduce false positives.  
 



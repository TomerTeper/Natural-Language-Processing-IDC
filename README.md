# NLP Projects - M.Sc. Machine Learning & Data Science  
**Author:** Tomer Teperovich  
**Program:** M.Sc. in Machine Learning & Data Science, Reichman University  

---

## Assignment 1 – Character-Based N-gram Language Models  
**Goal:** Build and evaluate character-level N-gram language models for multilingual tweet datasets (8 Latin-script languages).  

**Highlights:**
- Implemented unsmoothed and add-one smoothed N-gram models (1–4 grams).  
- Evaluated cross-language **perplexity matrices** to identify native language fit.  
- Generated text samples based on learned character distributions.  

**Key topics:** tokenization · probability modeling · smoothing · perplexity evaluation · text generation  
📘 [Read full README](HW1_100/README.md)

---

## Assignment 2 – Named Entity Recognition 
**Goal:** Develop and compare multiple **LSTM-based NER models** using the ReCoNLL-2003 dataset.  

**Highlights:**
- Trained **unidirectional and bidirectional LSTMs** with learned and pretrained **GloVe embeddings**.  
- Achieved **F1-score = 0.95** (w/o O-tags) using BiLSTM(3) + GloVe-300d.  
- Performed detailed **error analysis** and confusion matrix evaluation.  

**Key topics:** sequence labeling · LSTM/BiLSTM · embeddings · precision/recall/F1  
📘 [Read full README](HW2_100/README.md)

---

## NLP Final Project – Hebrew Multi-Word Expressions (Idioms)  
**Goal:** Examine how different Transformer models handle **Hebrew idioms** through masked word prediction and generative prompting.  

**Highlights:**
- Compared **AlephBERT**, **mBERT**, and **GPT-4o-mini** on idiomatic vs literal sentences.  
- Introduced **Fuzzy matching** evaluation to capture partial lexical accuracy.  
- Achieved **78.7 % Top-5 Fuzzy** (AlephBERT) and **65.9 % CoT-Prompt GPT-4o-mini**.  

**Key topics:** MWEs · BERT masking · few-shot prompting · Hebrew NLP  
📘 [Read full README](Final_project_91/README.md)

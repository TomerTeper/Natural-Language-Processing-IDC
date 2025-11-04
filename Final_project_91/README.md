# Identifying and Completing Hebrew Multi-Word Expressions  
### Comparative Evaluation of Masked and Generative Language Models

**Course:** Natural Language Processing (M.Sc. ML & DS - Reichman University, 2025)  

---

## Overview
This project investigates how modern Hebrew and multilingual Transformer models understand **Multi-Word Expressions (MWEs)** - especially idioms whose meanings are **non-compositional**.  
The task is formulated as **Masked Word Completion**: given a Hebrew sentence containing a masked idiom component, models must predict the correct missing word.

Three architectures were compared:

| Model | Type | Training | Strength |
|:------|:------|:----------|:----------|
| **AlephBERT** | Encoder (BERT) | Hebrew corpus | Strong morphology & collocations |
| **mBERT** | Encoder (BERT) | 100 languages | Weak idiom sensitivity |
| **GPT-4o-mini** | Decoder LLM | Prompt-based (0/2/4/8-shot + CoT) | Semantic reasoning |

---

## Dataset
**Source:** Hebrew subset of the [PARSEME 2020](https://parsemefr.lis-lab.fr) corpus (`train.cupt`, 14 035 sentences).  
**Extraction pipeline:**
1. Identify all tokens marked as MWEs (column 11 = `PARSEME:MWE`).  
2. Select ~47 idiomatic sentences (≥ 2 tokens).  
3. Mask the final idiom token (`[…] → [MASK]`).  
4. Add 18 manual **literal controls** where the same lexemes appear in non-idiomatic contexts.

Example pair:  

| Type | Sentence | Gold word |
|:--|:--|:--|
| Idiom | _הוועדה תמשיך לעקוב עם יד על [MASK]._ | הדופק |
| Literal | _האחות בדקה אותי עם יד על [MASK]._ | הדופק |

---

## Methodology

### Data Processing
- Parse `.cupt` files → extract token sequences + MWE metadata  
- Build DataFrame (`original`, `masked`, `literal`) → save `masked_mwe_sentences.csv`  

### Model Evaluation
- **AlephBERT / mBERT:** Hugging Face `fill-mask` pipeline  
- **GPT-4o-mini:** prompted via OpenAI API (Zero-, Few-, CoT-shot)  
- Evaluate per sentence → Top-1 / Top-5 Exact & Fuzzy metrics  

### Metrics
| Metric | Description |
|:--|:--|
| **Top-1 Exact** | Correct word = top prediction |
| **Top-1 Fuzzy** | Levenshtein similarity ≥ 85 % |
| **Top-5 Exact/Fuzzy** | Gold word appears within Top 5 |

---

## Results

### Overall Model Performance
| Model | Top-1 Exact % | Top-1 Fuzzy % | Top-5 Exact % | Top-5 Fuzzy % |
|:--|:--:|:--:|:--:|:--:|
| **AlephBERT** | 61.7 | 68.1 | 76.6 | **78.7** |
| **GPT-4o-mini** | 34.0 | 46.8 | 59.6 | 66.0 |
| **mBERT** | 10.6 | 10.6 | 12.8 | 12.8 |

🔹 **AlephBERT** dominates across all metrics → 78.7 % Top-5 Fuzzy.  
🔹 **mBERT** shows weak performance → poor Hebrew idiom coverage.  
🔹 **GPT-4o-mini** benefits greatly from prompt engineering.

---

### Effect of Prompting on GPT-4o-mini
| Prompt Method | Top-5 Exact % | Top-5 Fuzzy % |
|:--|:--:|:--:|
| Zero-Shot | 53.2 | 57.5 |
| Few-Shot (2) | 46.8 | 53.2 |
| Few-Shot (4) | 55.3 | 61.7 |
| Few-Shot (8) | 51.1 | 59.6 |
| MWE-Informed Prompt | 57.5 | 65.9 |
| Chain-of-Thought (C o T) | 59.6 | 65.9 |

**Few-Shot (4)** + **CoT** ≈ best balance between accuracy & reasoning.

---

### Idiomatic vs Literal Sentences
| Model | Idiomatic Fuzzy % | Literal Fuzzy % |
|:--|:--:|:--:|
| **AlephBERT** | 68.1 | 66.7 |
| **GPT-4o-mini** | 38.3 | 55.6 |
| **mBERT** | 10.6 | 5.6 |

> Encoders (BERTs) memorize fixed idioms → higher accuracy on idiomatic cases.  
> GPT shows more semantic flexibility → better literal handling.

---

## Key Findings
1. **Language-specific pretraining wins:** AlephBERT best captures Hebrew morphology and collocations.  
2. **Prompt engineering matters:** Few-Shot (+ CoT) significantly improves LLM performance.  
3. **“Curse of Context”:** models perform better on fixed idioms than on literal contexts.  
4. **Complementary architectures:** Encoders = collocational memory; Decoders = semantic reasoning.

---

## Key Topics Covered
- Multi-Word Expressions (MWEs) and idioms in Hebrew  
- Transformer Masked Language Models (BERT family)  
- Generative prompting (GPT-4o-mini Few-Shot / CoT)  
- Exact vs Fuzzy matching evaluation  
- Statistical collocations vs semantic reasoning  

---

## Future Work
- Extend literal dataset and mask non-final idiom positions.  
- Add more decoder-only LLMs (Gemini, Claude).  
- Explore self-consistency (CoT + SC) prompting.  
- Generate synthetic idioms via LLM data augmentation.


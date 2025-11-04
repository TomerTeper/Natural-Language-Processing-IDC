# Assignment 1 – Character-Based N-gram Language Models  

## Overview
This project implements a complete pipeline for building and evaluating **character-based N-gram language models**.  
The main goal is to **detect the language of tweets** in eight different Latin-script languages by modeling the probability distribution of character sequences.

The notebook develops reusable tools for:
- Vocabulary creation from multilingual corpora  
- Building smoothed and unsmoothed N-gram language models  
- Computing model **perplexity** for language evaluation  
- Cross-language comparison via **perplexity matrix**  
- Generating new text sequences using learned language models  

---

## Key Topics Covered
- Character-level tokenization  
- N-gram modeling (1- to 4-grams)  
- Add-one (Laplace) smoothing  
- Perplexity computation  
- Text generation from probabilistic models  
- Cross-language evaluation and comparison  

---

## Dataset
Tweets in eight Latin-alphabet languages:
| Code | Language |
|------|-----------|
| en | English |
| es | Spanish |
| fr | French |
| in | Indonesian |
| it | Italian |
| nl | Dutch |
| pt | Portuguese |
| tl | Tagalog |

Each file (e.g., `en.csv`, `fr.csv`, …) contains tweets under the column **`tweet_text`**.  

---

## Methodology

### 1. Preprocessing (`preprocess`)
Builds a **shared character vocabulary** across all datasets:
- Iterates through every CSV file in `/data`
- Extracts all UTF-8 characters seen in tweets
- Adds special tokens `<start>` and `<end>`

### 2. Language Model Construction (`build_lm`)
Creates a probabilistic **N-gram model**:
- Counts all N-grams in each tweet with proper start/end padding  
- Supports **add-one smoothing** and `<unk>` token handling  
- Outputs a nested dictionary:
  ```python
  LM[context][next_char] = P(next_char | context)
  ```

### 3. Evaluation (`eval` and `perplexity`)
Measures how well a model predicts unseen text:
- Calculates **perplexity** for a given test language
- Handles unknown characters via `<unk>`
- Averages perplexity over all tweets in the dataset

### 4. Cross-Language Comparison (`match`)
Builds a **perplexity matrix** across all 8×8 language pairs for `n ∈ {1,2,3,4}`:
| source | target | n | perplexity |
|---------|---------|---|------------|
| en | en | 3 | 123.4 |
| en | fr | 3 | 241.5 |
| … | … | … | … |

Helps verify that models perform best on their native language.

### 5. Text Generation (`generate`)
Generates text from the learned model:
- Starts with a **prompt**
- Samples the next character according to learned probabilities
- Stops when `<end>` token is reached or after `N` tokens

Example:
```python
print(generate('en', 3, "I am", 10, 5))
# Output: I am ver...
```

---

## Experiments & Key Findings
- Models with **higher n-grams (n=3,4)** yield **lower perplexity** on the same language.
- Cross-language perplexity confirms that models generalize poorly across languages, proving that character-based models capture language-specific patterns.
- Add-one smoothing significantly stabilizes probabilities and reduces unseen-token errors.




# generative_ai_sanobar_scfu123012

# activity -1

exploring llm models on hugging face

# Activity 2 — Tokenization Practical

## Overview

This activity explores how BERT's tokenizer breaks text into tokens using the Hugging Face `transformers` library. The goal is to understand how subword tokenization works and why it matters for language models.

## What This Notebook Covers

- Loading the `bert-base-uncased` tokenizer
- Tokenizing single words, names, and full sentences
- Converting tokens into token IDs
- Observing special tokens (`[CLS]`, `[SEP]`) added automatically during encoding

## Setup

```bash
pip install transformers
```

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

## Key Questions Explored

**Is one word always one token?**
No. Common words (e.g. "hello") are usually a single token. Longer, rarer, or made-up words get split into subword pieces (e.g. "unbelievable" → `un`, `##believ`, `##able`).

**Why are some words split?**
BERT uses **WordPiece** tokenization with a fixed vocabulary (~30,000 entries). Words not in the vocabulary are broken into smaller known chunks so the model never encounters a completely unknown word.

**What happens to punctuation?**
Punctuation marks are tokenized separately from the words they're attached to (e.g. "amazing." → `amazing`, `.`).

**Why does the model need token IDs?**
Neural networks operate on numbers, not text. Each token maps to a unique integer ID in the vocabulary — this is the numeric input the model actually processes.

## Sample Results

| Text      | Tokens                    | Notes                               |
| --------- | ------------------------- | ----------------------------------- |
| `sanobar` | `['san', '##oba', '##r']` | Proper nouns/names are often split  |
| `january` | `['january']`             | Common word, single token           |
| `5`       | `['5']`                   | Numbers are typically single tokens |
| `example` | `['example']`             | Common word, single token           |

## Tech Stack

- Python 3.13
- Hugging Face `transformers` (v5.15.0)
- Jupyter Notebook (`activity-2.ipynb`)

## Author

Sanobar — B.Tech CSE, MIT Vishwaprayag University

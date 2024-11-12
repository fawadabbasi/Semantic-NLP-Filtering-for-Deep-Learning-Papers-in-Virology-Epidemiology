# Virology Papers Classification using SciBERT 
This project utilizes BERT embeddings and cosine similarity to classify virology research papers based on keywords and relevant classification terms. 
The code extracts relevant abstracts, preprocesses them, and then classifies them into different categories based on natural language processing and computer vision methods.
## Features
- **Data Preprocessing:** Remove stopwords, lowercase text, and filter out abstracts without content.
- **Keyword Matching and Classification:** Use a predefined set of keywords for matching relevant research topics.
- **BERT Embeddings:** Extract embeddings using `allenai/scibert_scivocab_uncased` for text similarity calculations.
- **Cosine Similarity Scoring:** Rank abstracts based on relevance to predefined terms.
- **Classification of Topics:** Classify papers into "text mining," "computer vision," "both," or "other" categories.
## Requirements
To install the required dependencies, run:
```bash
pip install transformers scikit-learn torch pandas nltk
Additional setup for NLTK stopwords:
```bash
import nltk
nltk.download('stopwords')

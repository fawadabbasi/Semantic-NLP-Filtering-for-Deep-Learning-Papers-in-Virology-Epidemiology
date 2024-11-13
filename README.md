# Virology Papers Classification using SciBERT 
This project utilizes BERT embeddings and cosine similarity to classify virology research papers based on keywords and relevant classification terms. 
The code extracts relevant abstracts, preprocesses them, and then classifies them into different categories based on natural language processing and computer vision methods.
## Features
- **Data Preprocessing:** Remove stopwords, lowercase text, and filter out abstracts without content.
- **Keyword Matching and Classification:** Use a predefined set of keywords for matching relevant research topics.
- **BERT Embeddings:** Extract embeddings using `allenai/scibert_scivocab_uncased` for text similarity calculations.
- **Cosine Similarity Scoring:** Rank abstracts based on relevance to predefined terms.
- **Classification of Topics:** Classify papers into "text mining," "computer vision," "both," or "other" categories.
  
## GPU Usage for Optimization
The code was optimized using GPU acceleration for faster embedding generation and processing. 

## Requirements
To install the required dependencies, run:
```bash
pip install transformers scikit-learn torch pandas nltk
```
Additional setup for NLTK stopwords:
```
import nltk
nltk.download('stopwords')
```
## Code Walkthrough
- **Loading and Preprocessing Data:** Load abstracts from collection_with_abstracts.csv, filter out empty entries, and apply text preprocessing.
- **Embedding Generation:** Use SciBERT to generate embeddings for abstracts and keywords.
- **Cosine Similarity Computation:** Compare embeddings and compute relevance scores.
- **Classification and Keyword Extraction:** Classify papers based on content and relevant terms.
- **Save Results:** Filter and save the results with classification labels and relevance scores to filtered_virology_papers_bert.csv.
## Usage
To use this code, ensure your data file (collection_with_abstracts.csv) is in the working directory. You can then execute the code to produce a CSV file with classifications and method names for each abstract.

## Files
- collection_with_abstracts.csv: Input file containing abstracts for processing.
- filtered_virology_papers_bert.csv: Output file with filtered abstracts and classifications.
  
## Why SciBERT and Cosine Similarity?
Traditional keyword-based filtering lacks contextual understanding  it may include papers using irrelevant terms.SciBERT embeddings encode both the semantic and syntactic structure of the text. Using cosine similarity on these embeddings allows the system to detect related topics even when keywords are absent, providing an  accurate classification.

## Why SciBERT and not BERT?
SciBERT leverages unsupervised pretraining BERT on a large multi-domain corpus of scientific publications to improve performance on downstream scientific NLP tasks.

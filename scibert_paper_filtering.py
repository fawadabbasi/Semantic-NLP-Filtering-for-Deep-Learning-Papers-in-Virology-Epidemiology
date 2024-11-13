from transformers import AutoModel, AutoTokenizer, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import torch

# Load dataset using pandas
data = pd.read_csv('collection_with_abstracts.csv', nrows=12000)

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Keywords and classification terms
keywords = [
    'convolutional neural network', 'cnn', 'rnn', 'lstm', 'u net', 
    'transfer learning', 'bert', 'transformer', 'resnet', 
    'graph neural network', 'aggregation network', 'gnn', 'med bert', 'bio bert', 'sci bert',
    'machine learning', 'deep learning', 'neural network'
]

classification_terms = {
    "text mining": ["text mining", "nlp", "natural language processing", "bert", "bio bert", "sci bert",
                    'text analysis', 'text classification', "med bert"],
    "computer vision": ["image processing", "image recognition", "computer vision", "cnn", "convolutional", "u net"],
    "both": ["both", "multi modal", "hybrid"]
}

# Filter out rows with empty abstracts
data = data.dropna(subset=['abstract'])
data = data[data['abstract'].str.strip() != ""]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'\W+', ' ', text)  # remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words and len(word) > 2])  # remove stopwords
    return text

# Apply preprocessing
data['processed_abstract'] = data['abstract'].apply(preprocess_text)

# Set device for GPU usage
device = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
nlp_pipeline = pipeline(
    'feature-extraction', 
    model=model, 
    tokenizer=tokenizer, 
    device=device, 
    max_length=512, 
    truncation=True
)

# Get BERT embeddings in batches
# Tokenize and truncate, then pass to the model
def get_bert_embeddings(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Explicitly tokenize with truncation
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"  # Return PyTorch tensors
        )
        encoded_inputs = encoded_inputs.to(device)  # Move inputs to the GPU

        with torch.no_grad():
            model_outputs = model(**encoded_inputs)
            batch_embeddings = model_outputs.last_hidden_state.mean(dim=1)  # Mean pooling over sequence length
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.vstack(embeddings)  # Stack all batches into a single numpy array


# Obtain embeddings for the abstracts and keywords
abstract_embeddings = get_bert_embeddings(data['processed_abstract'].tolist())
keyword_embeddings = get_bert_embeddings(keywords)

# Compute cosine similarity and classify papers
def classify_paper(text):
    for method, terms in classification_terms.items():
        if any(term in text for term in terms):
            return method
    return "other"

def extract_method_name(text):
    for term in keywords:
        if term in text:
            return term
    return 'unknown'

# Compute relevance scores and classify papers
relevance_scores = cosine_similarity(abstract_embeddings, keyword_embeddings).max(axis=1)
data['relevance_score'] = relevance_scores
data = data[data['relevance_score'] >= 0.6]

# Apply classification and method extraction
data['classification'] = data['processed_abstract'].apply(classify_paper)
data['method_name'] = data['processed_abstract'].apply(extract_method_name)

# Save the filtered and classified results to a CSV file
data[['PMID', 'abstract', 'classification', 'method_name']].to_csv('filtered_virology_papers_bert.csv', index=False)

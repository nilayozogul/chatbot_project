import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss

# Embedding modeli
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def load_dataset(excel_path):
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()  # Kolon başlıklarındaki boşlukları temizle
    print("Kolonlar:", df.columns.tolist())  # Debug için
    df.dropna(subset=['user_input', 'intent', 'answer'], inplace=True)
    return df

def create_embeddings(texts):
    model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return model, embeddings

def save_vector_store(embeddings, df, filename='vector_store.pkl'):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Get directory of the pickle file to save faiss.index next to it
    base_dir = os.path.dirname(filename)
    index_name = "faiss.index"
    index_path = os.path.join(base_dir, index_name)
    
    faiss.write_index(index, index_path)
    with open(filename, 'wb') as f:
        pickle.dump({'index': index_name, 'df': df}, f)

def load_vector_store(filename='vector_store.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Get directory of the pickle file to find faiss.index
    base_dir = os.path.dirname(filename)
    index_path = os.path.join(base_dir, data['index'])
    
    index = faiss.read_index(index_path)
    return index, data['df'] 
import os
import openai
import pandas as pd
import pickle
import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
sys.path.append('..')
from rag_utils import load_dataset, create_embeddings, save_vector_store, load_vector_store

class LlamaModel:
    def __init__(self, api_key_path="API_KEY.txt", data_path="../data/data_set.xlsx", 
                 vector_store_path="vector_store.pkl", intent_model_path="intent_classifier.pkl"):
        # API Key yükle
        with open(api_key_path, "r") as f:
            lines = f.readlines()
            api_key = None
            for line in lines:
                if line.startswith("LLAMA_API_KEY="):
                    api_key = line.strip().split("=")[1]
                    break
            if not api_key:
                raise ValueError("LLAMA_API_KEY bulunamadı!")
        
        openai.api_key = api_key
        openai.api_base = "https://openrouter.ai/api/v1"
        
        # Veri seti ve embedding yükle
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        self.intent_model_path = intent_model_path
        
        # Load vector store
        if not os.path.exists(vector_store_path):
            print("Vector store bulunamadı. Yeniden oluşturuluyor...")
            df = load_dataset(data_path)
            embedder, embeddings = create_embeddings(df['user_input'].tolist())
            save_vector_store(embeddings, df, vector_store_path)
            print("Vector store oluşturuldu!")
        
        self.index, self.df = load_vector_store(vector_store_path)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Load intent classifier
        self.intent_clf = joblib.load(intent_model_path)
    
    def predict_intent(self, user_question):
        """Kullanıcı sorusunun intent'ini tahmin eder"""
        user_emb = self.embedder.encode([user_question])
        intent_probs = self.intent_clf.predict_proba(user_emb)[0]
        intent_idx = np.argmax(intent_probs)
        intent_pred = self.intent_clf.classes_[intent_idx]
        confidence = intent_probs[intent_idx] * 100
        return intent_pred, confidence
    
    def get_classifier_response(self, intent_pred, confidence, fallback_threshold=70.0):
        """Intent classifier'a göre cevap döner"""
        if confidence >= fallback_threshold:
            df_intent = self.df[self.df['intent'] == intent_pred]
            if not df_intent.empty:
                return df_intent.sample(1)['answer'].values[0]
        return "Sorry, I didn't quite understand your question."
    
    def get_rag_response(self, user_question):
        """RAG kullanarak cevap döner"""
        user_emb = self.embedder.encode([user_question])
        D, I = self.index.search(user_emb, k=3)
        retrieved_rows = self.df.iloc[I[0]]
        
        # Build RAG prompt
        prompt = "You are a helpful customer support chatbot. Here are some similar answers from our knowledge base:\n\n"
        for idx, row in retrieved_rows.iterrows():
            prompt += f"Intent: {row['intent']}\nUser Input: {row['user_input']}\nAnswer: {row['answer']}\n\n"
        
        prompt += f"\nCustomer question: {user_question}\n\n"
        prompt += "Please respond based on the above answers. Be helpful, polite, and avoid generating new information."
        
        # API call
        response = openai.ChatCompletion.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        return response['choices'][0]['message']['content'].strip()
    
    def get_response(self, user_question, use_classifier_only=False):
        """Ana response fonksiyonu"""
        # Intent prediction
        intent_pred, confidence = self.predict_intent(user_question)
        
        # Classifier response
        classifier_response = self.get_classifier_response(intent_pred, confidence)
        
        # RAG response (if enabled)
        if not use_classifier_only and confidence >= 70.0:
            rag_response = self.get_rag_response(user_question)
            return rag_response, intent_pred, confidence
        else:
            return classifier_response, intent_pred, confidence 
# response_evaluate.py

import pandas as pd
import random
import time
from models.llama_model import LlamaModel
from models.gemini_model import GeminiModel

# Init models
llama_model = LlamaModel(
    api_key_path="API_KEY.txt",
    data_path="data/data_set.xlsx",
    vector_store_path="models/vector_store.pkl",
    intent_model_path="models/intent_classifier.pkl"
)

gemini_model = GeminiModel(
    api_key_path="API_KEY.txt",
    data_path="data/data_set.xlsx",
    vector_store_path="models/vector_store.pkl",
    intent_model_path="models/intent_classifier.pkl"
)

# Load dataset
df = pd.read_excel("data/data_set.xlsx")
df.columns = df.columns.str.strip()
df.dropna(subset=['user_input', 'intent', 'answer'], inplace=True)

# Sample random test inputs
num_samples = 10
df_sample = df.sample(n=num_samples, random_state=42)

# Run responses
for idx, row in df_sample.iterrows():
    user_input = row['user_input']
    ground_truth = row['answer']

    print(f"\n=== Sample {idx+1}/{num_samples} ===")
    print(f"User Input    : {user_input}")
    print(f"Ground Truth  : {ground_truth}\n")

    # LLaMA response
    try:
        llama_reply, intent_pred_llama, conf_llama = llama_model.get_response(user_input)
        print(f"LLaMA Response: {llama_reply}\n")
    except Exception as e:
        print(f"LLaMA API ERROR: {e}")

    # Gemini response
    try:
        gemini_reply, intent_pred_gemini, conf_gemini = gemini_model.get_response(user_input)
        print(f"Gemini Response: {gemini_reply}\n")
    except Exception as e:
        print(f"Gemini API ERROR: {e}")

    print("="*50)
    time.sleep(3)  # avoid hitting rate limits

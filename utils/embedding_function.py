import torch
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
import os

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")

MODEL_ID = "google/embeddinggemma-300m" 

print(f"Loading {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    token=hf_token
)

model = AutoModel.from_pretrained(
    MODEL_ID, 
    device_map="cpu", 
    token=hf_token
)

def embedding_function(text):
    """
    Generates a vector embedding for the given text using the Gemma model.
    
    Args:
        text (str): The input text to embed.
        
    Returns:
        list: A list of floats representing the embedding vector.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    attention_mask = inputs['attention_mask']
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    embedding = sum_embeddings / sum_mask

    return embedding[0].cpu().tolist()

def focussed_embedding_function(text, query):
    """
    Splits text into sentences, finds the sentence most similar to the query 
    using the Gemma model, and returns that specific sentence's embedding.

    Args:
        text (str): The full input text.
        query (str): The search query to focus the embedding.

    Returns:
        list: The embedding vector of the best matching sentence.
    """
    
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        print("Warning: No sentences found in text. returning query embedding instead.")
        return embedding_function(query)

    query_vec = embedding_function(query)
    
    query_vec_np = np.array(query_vec)
    
    best_score = -1.0
    best_embedding = None
    best_sentence_text = ""

    for sentence in sentences:
        sent_vec = embedding_function(sentence)
        sent_vec_np = np.array(sent_vec)

        dot_product = np.dot(query_vec_np, sent_vec_np)
        norm_query = np.linalg.norm(query_vec_np)
        norm_sent = np.linalg.norm(sent_vec_np)
        
        if norm_sent == 0 or norm_query == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_query * norm_sent)

        if similarity > best_score:
            best_score = similarity
            best_embedding = sent_vec
            best_sentence_text = sentence

    
    return best_embedding, best_sentence_text


if __name__ == "__main__":
    text = "This is a sample text. It contains multiple sentences. The goal is to find the most relevant sentence."
    print("General Embedding:", embedding_function(text))
    query = "relevant sentence"
    best_embedding, best_sentence_text = focussed_embedding_function(text, query)
    print("Best Sentence:", best_sentence_text)
import pandas as pd
import numpy as np

from tqdm import tqdm
import chromadb
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from utils.embedding_function import embedding_function
from utils.embedding_function import focussed_embedding_function
from utils.load_data import clean_format_dataframe,folder_to_dataframe


def encode_data(clean_data_frame):
    """
    Encodes text data into feature vectors and tags into one-hot encoded labels.



    This function takes a DataFrame, concatenates specified text columns,

    and uses a provided embedding function to create a feature matrix 'X'.

    It also converts the 'tags' column into a one-hot encoded matrix 'y'

    for multi-label classification.



    Args:

        clean_data_frame (pd.DataFrame): The DataFrame containing the cleaned data.

                                         It must include the text columns and a 'tags' column.

        embedding_function (callable): A function that takes a list of strings

                                       and returns their numerical embeddings as a numpy array.



    Returns:

        tuple: A tuple containing:

            - np.ndarray: The feature matrix (X) of shape (n_samples, n_features).

            - np.ndarray: The one-hot encoded label matrix (y) of shape (n_samples, n_classes).

    """
    clean_data_frame['prob_desc_notes'] = clean_data_frame['prob_desc_notes'].fillna('')
    clean_data_frame['prob_desc_output_spec'] = clean_data_frame['prob_desc_output_spec'].fillna('')
    clean_data_frame['prob_desc_input_spec'] = clean_data_frame['prob_desc_input_spec'].fillna('')
    clean_data_frame['prob_desc_description'] = clean_data_frame['prob_desc_description'].fillna('')

    text_columns = ['prob_desc_input_spec', 'prob_desc_output_spec', 'prob_desc_description', 'prob_desc_notes']
    text_to_encode = clean_data_frame[text_columns].agg('\n'.join, axis=1).tolist()
    print(text_to_encode)
    
    print(f'Total samples to encode: {len(text_to_encode)}')

    list_of_embeddings = []
    
    for text in tqdm(text_to_encode, desc="Encoding individually"):
        
        single_embedding = embedding_function([text])
        
        if isinstance(single_embedding, np.ndarray):
            single_embedding = single_embedding.flatten()
        elif isinstance(single_embedding, list):
            single_embedding = np.array(single_embedding).flatten()
            
        list_of_embeddings.append(single_embedding)
    
    X = np.vstack(list_of_embeddings)
    
    print(f"DEBUG: X shape is {X.shape}")
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")
    collection = chroma_client.get_or_create_collection(name="code_problems")

    ids = [str(i) for i in range(len(clean_df))]
    collection.add(
        embeddings=X.tolist(),
        documents=text_to_encode,
        ids=ids
    )

    return X

def get_labels(clean_data_frame):
    """
    Extracts one-hot encoded labels from the cleaned DataFrame.

    Args:
        clean_data_frame (pd.DataFrame): The DataFrame containing the cleaned data.
                                         It must include the one-hot encoded tag columns.

    Returns:
        np.ndarray: The one-hot encoded label matrix (y) of shape (n_samples, n_classes).
    """
    tag_columns = [col for col in clean_data_frame.columns if col.startswith('tag_')]
    y = clean_data_frame[tag_columns].values

    return y

def encode_data_focussed(clean_data_frame, embedding_function):
    """
    Encodes data by finding the specific sentence in each document 
    that best matches a set of predefined tags.
    """
    text_columns = ['prob_desc_input_spec', 'prob_desc_output_spec', 'prob_desc_description', 'prob_desc_notes']
    clean_data_frame[text_columns] = clean_data_frame[text_columns].fillna('')
    
    text_to_encode = clean_data_frame[text_columns].agg('\n'.join, axis=1).tolist()
    
    tags = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']

    chroma_client = chromadb.PersistentClient(path="./my_vector_db_focussed")
    try:
        chroma_client.delete_collection("code_problems_focussed")
    except :
        pass 
    collection = chroma_client.create_collection(name="code_problems_focussed")

    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []

    print(f"Processing {len(text_to_encode)} documents against {len(tags)} tags...")

    for doc_idx, text in tqdm(enumerate(text_to_encode), total=len(text_to_encode), desc="Focussed Encoding"):
        
        for tag in tags:
            best_emb, best_sent = focussed_embedding_function(text, tag)
            
            unique_id = f"{doc_idx}_{tag}"
            
            batch_ids.append(unique_id)
            batch_embeddings.append(best_emb)
            batch_documents.append(best_sent)
            batch_metadatas.append({
                "original_doc_id": doc_idx,
                "matched_tag": tag
            })

    if batch_ids:
        chunk_size = 5000
        for i in range(0, len(batch_ids), chunk_size):
            collection.add(
                ids=batch_ids[i : i+chunk_size],
                embeddings=batch_embeddings[i : i+chunk_size],
                documents=batch_documents[i : i+chunk_size],
                metadatas=batch_metadatas[i : i+chunk_size]
            )
            
    return None

def load_embeddings_from_chroma(db_path="./my_vector_db", collection_name="code_problems"):
    print("Loading from ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=collection_name)
    
    result = collection.get(include=['embeddings', 'documents'])
    
    ids = result['ids']
    embeddings = result['embeddings']
    documents = result['documents']
    
    combined = list(zip(ids, embeddings, documents))
    
    combined_sorted = sorted(combined, key=lambda x: int(x[0]))
    
    sorted_ids, sorted_embeddings, sorted_documents = zip(*combined_sorted)
    
    X = np.array(sorted_embeddings)
    
    print(f"Loaded X with shape: {X.shape}")
    print(f"First ID: {sorted_ids[0]}, Last ID: {sorted_ids[-1]}")
    
    return X, sorted_documents

def load_embeddings_focussed(db_path="./my_vector_db_focussed", collection_name="code_problems_focussed"):
    print("Loading from ChromaDB Focussed...")
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_collection(name=collection_name)
    
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    embeddings = result['embeddings']
    documents = result['documents']
    metadatas = result['metadatas']
    
    if embeddings is None or len(embeddings) == 0:
        print("Database is empty.")
        return np.array([]), []

    tags_order = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
    tag_to_idx = {tag: i for i, tag in enumerate(tags_order)}
    n_tags = len(tags_order)

    all_doc_ids = [int(m['original_doc_id']) for m in metadatas] # Ensure ints
    n_docs = max(all_doc_ids) + 1
    
    embedding_dim = len(embeddings[0])
    
    X = np.zeros((n_docs, embedding_dim, n_tags))
    
    sorted_documents = [{} for _ in range(n_docs)]

    print(f"Constructing matrix of shape: {X.shape}...")

    for emb, doc_text, meta in zip(embeddings, documents, metadatas):
        doc_id = int(meta['original_doc_id'])
        tag = meta['matched_tag']
        
        if tag in tag_to_idx:
            tag_idx = tag_to_idx[tag]
            
            X[doc_id, :, tag_idx] = emb
            
            sorted_documents[doc_id][tag] = doc_text

    print(f"Loaded X with shape: {X.shape}")
    

    return X, sorted_documents

if __name__ == "__main__":
    raw_df = folder_to_dataframe('../code_classification_dataset')
    clean_df = clean_format_dataframe(raw_df)
    
    # X = encode_data(clean_df)
    # print(f"Feature matrix shape: {X.shape}")
    # X, sorted_documents = load_embeddings_from_chroma(db_path="./my_vector_db", collection_name="code_problems")
    
    encode_data_focussed(clean_df.head(20), focussed_embedding_function)
    X_focussed, sorted_documents_focussed = load_embeddings_focussed(db_path="./my_vector_db_focussed", collection_name="code_problems_focussed")
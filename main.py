import torch
import os
import sys
import argparse
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from utils.transform_data import clean_format_dataframe, folder_to_dataframe
    from classification_with_embeddings.neural_network_for_classification import MultiLabelClassifier, get_labels,evaluate_model_metrics,plot_test_metrics

    from utils.transform_data import encode_data
except ImportError:
    print("Error: Could not import from 'utils.transform_data'. Ensure the 'utils' folder is in the same directory.")
    sys.exit(1)

def predict(model, embeddings, threshold=0.5, class_names=None):
    model.eval()
    with torch.no_grad():
        probs = model(embeddings)
    
    # Convert probabilities to binary predictions
    preds = (probs > threshold).int().numpy()
    probs = probs.numpy()
    
    results = []
    for i in range(len(preds)):
        row_pred = preds[i]
        row_probs = probs[i]
        
        # Get names of predicted classes
        predicted_labels = [class_names[j] for j, val in enumerate(row_pred) if val == 1]
        
        results.append({
            "predicted_labels": predicted_labels,
            "raw_probabilities": row_probs
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference with the trained MultiLabel Classifier.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing test text files.")
    parser.add_argument('--model_path', type=str, default='./NN_model/final_model.pth', help="Path to the saved .pth model file.")
    parser.add_argument('--hidden_size', type=int, default=None, help="The hidden_size used in the best trial (e.g., 128, 256). Leave empty if None.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Probability threshold for classification.")
    parser.add_argument('--output_dir', type=str, default='./inference_output', help="Directory to save individual JSON files.")
    
    args = parser.parse_args()

    CLASS_NAMES = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
    INPUT_SIZE = 768
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"--- Loading data from {args.data_dir} ---")
    try:
        raw_df = folder_to_dataframe(args.data_dir)
        clean_df = clean_format_dataframe(raw_df)
        print(f"Loaded {len(clean_df)} documents.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("--- Generating embeddings ---")
    try:
        embeddings = encode_data(clean_df,output_database_path="./my_vector_db_inference")
        print(f"Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return
    

    print(f"--- Loading model from {args.model_path} ---")
    model = MultiLabelClassifier(INPUT_SIZE, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(args.model_path))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Tip: Check if the 'hidden_size' argument matches the one used during training.")
        return

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    print("--- Running Inference ---")
    results = predict(model, embeddings, threshold=args.threshold, class_names=CLASS_NAMES)
    
    if raw_df.columns.contains('tags'):
        y = get_labels(clean_df) 
        y = torch.tensor(y, dtype=torch.int32)
        test_metrics_data = evaluate_model_metrics(model, embeddings, y, class_names=CLASS_NAMES)
        
        plot_test_metrics(test_metrics_data)
        
    output_df = raw_df.copy()
    output_df['predicted_tags'] = [r['predicted_labels'] for r in results]
    
    print("\n--- Sample Predictions ---")
    print(output_df['predicted_tags'].head(5))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Saving individual JSON files to '{output_dir}/' ---")

    for index, row in output_df.iterrows():
        record = row.to_dict()
        
        if 'filename' in record and record['filename']:
            base_name = os.path.splitext(os.path.basename(str(record['filename'])))[0]
            file_name = f"{base_name}.json"
        else:
            file_name = f"sample_{index}.json"
            
        file_path = os.path.join(output_dir, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save {file_name}: {e}")

    print(f"Successfully saved {len(output_df)} files.")

if __name__ == "__main__":
    main()
    
    

    

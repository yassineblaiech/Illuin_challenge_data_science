import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, classification_report
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
from collections import Counter

sns.set_theme(style="whitegrid")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.transform_data import clean_format_dataframe, folder_to_dataframe, load_embeddings_from_chroma, get_labels

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=None):
        super(MultiLabelClassifier, self).__init__()
        
        if hidden_size:
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
                nn.Sigmoid()
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, num_classes),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.network(x)

class TagsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train_for_tuning(params, X_train, y_train, X_val, y_val, input_size, num_classes):
    lr = params.get('lr', 0.001)
    batch_size = params.get('batch_size', 32)
    epochs = params.get('epochs', 5)
    hidden_size = params.get('hidden_size', None)
    
    model = MultiLabelClassifier(input_size, num_classes, hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TagsDataset(X_train, y_train)
    val_dataset = TagsDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss

def train_model(X_train, y_train, X_val, y_val, input_size, num_classes, epochs=5, batch_size=32, lr=0.001, hidden_size=None):
    model = MultiLabelClassifier(input_size, num_classes, hidden_size=hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TagsDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TagsDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_model_weights = None  

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict()) 
            print(f"  -> New best model found at epoch {epoch+1}!")

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        print("Restored model to best state found during training.")

    return model, train_losses, val_losses

def evaluate_model_metrics(model, X_test, y_test, threshold=0.5, class_names=None):
    """
    Evaluates model and returns a dictionary containing aggregate and per-class metrics.
    
    Args:
        model: PyTorch model
        X_test: Input features
        y_test: True labels
        threshold: Decision threshold for probabilities
        class_names: (Optional) List of string names for the classes
    """
    model.eval()
    
    with torch.no_grad():
        probs = model(X_test)
        
    y_pred = (probs > threshold).float().cpu().numpy()
    y_true = y_test.cpu().numpy()
    
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    h_loss = hamming_loss(y_true, y_pred)
    
    metrics_data = {
        "Global_F1_Micro": f1_micro,
        "Global_F1_Macro": f1_macro,
        "Global_Precision_Micro": precision_micro,
        "Global_Recall_Micro": recall_micro,
        "Global_Hamming_Loss": h_loss
    }

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    num_classes = len(f1_per_class)

    for i in range(num_classes):
        cls_label = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        
        metrics_data[f"{cls_label}_F1"] = f1_per_class[i]
        metrics_data[f"{cls_label}_Precision"] = prec_per_class[i]
        metrics_data[f"{cls_label}_Recall"] = rec_per_class[i]

    print("--- Global Evaluation Metrics ---")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"Hamming Loss:     {h_loss:.4f}")
    
    print("\n--- Per-Class Breakdown ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    return metrics_data

def evaluate_model_loss(model, X_test, y_test):
    model.eval()
    criterion = nn.BCELoss()
    test_dataset = TagsDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    total_loss = 0
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
    print(f"Final Test Set Loss: {total_loss / len(test_loader):.4f}")

def objective(trial):
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'hidden_size': trial.suggest_categorical('hidden_size', [None, 128, 256]),
        'epochs': trial.suggest_categorical('epochs', [10, 20, 30, 40]),
    }
    
    val_loss = train_for_tuning(params, X_train, y_train, X_val, y_val, 768, 8)
    
    return val_loss

def run_optuna_optimization():
    print("--- Starting Optuna Optimization ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5) 
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return trial.params
    

def plot_training_history(train_losses, val_losses):
    """
    Plots the training and validation loss over epochs using matplotlib.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    
    plt.title('Training and Validation Loss across Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (BCE)', fontsize=12)
    plt.legend(fontsize=12)
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    print("Showing training history plot...")
    plt.show()

def plot_test_metrics(metrics_data):
    """
    Plots the final test metrics:
    1. Bar chart for Global Metrics.
    2. Table for Per-Class Metrics.
    """
    global_data = {}
    class_data_raw = {}
    
    std_aggregates = ["F1 Micro", "F1 Macro", "Precision", "Recall", "Hamming Loss", 
                      "Global_F1_Micro", "Global_F1_Macro", "Global_Precision_Micro", 
                      "Global_Recall_Micro", "Global_Hamming_Loss"]

    for k, v in metrics_data.items():
        if k.startswith("Global") or k in std_aggregates:
            global_data[k] = v
        else:
            class_data_raw[k] = v

    parsed_data = {}
    suffixes = ["_F1", "_Precision", "_Recall"]
    
    for k, v in class_data_raw.items():
        matched = False
        for suffix in suffixes:
            if k.endswith(suffix):
                class_name = k[:-len(suffix)]  
                metric_name = suffix.strip('_') 
                
                if class_name not in parsed_data:
                    parsed_data[class_name] = {}
                parsed_data[class_name][metric_name] = v
                matched = True
                break
    
    if parsed_data:
        df_class = pd.DataFrame.from_dict(parsed_data, orient='index')
        cols = [c for c in ["F1", "Precision", "Recall"] if c in df_class.columns]
        df_class = df_class[cols]
    else:
        df_class = pd.DataFrame()

    fig_height = 8 + (len(df_class) * 0.3) if not df_class.empty else 6
    fig, axes = plt.subplots(2, 1, figsize=(10, fig_height), gridspec_kw={'height_ratios': [1, 1]})

    if global_data:
        names = list(global_data.keys())
        values = list(global_data.values())
        
        clean_names = [n.replace("Global_", "").replace("_", " ") for n in names]
        
        sns.barplot(x=clean_names, y=values, palette="viridis", ax=axes[0])
        axes[0].set_title('Global Evaluation Metrics', fontsize=16)
        axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis='x', rotation=15)
        
        for i, v in enumerate(values):
            axes[0].text(i, v + 0.02, f"{v:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        axes[0].text(0.5, 0.5, "No Global Metrics Found", ha='center', fontsize=12)
        axes[0].axis('off')

    axes[1].axis('off') 
    
    if not df_class.empty:
        table_data = []
        headers = ["Class"] + list(df_class.columns)
        
        for index, row in df_class.iterrows():
            formatted_row = [f"{x:.4f}" for x in row]
            table_data.append([index] + formatted_row)
            
        table = axes[1].table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        axes[1].set_title('Per-Class Metrics Breakdown', fontsize=16, pad=20)
    else:
        axes[1].text(0.5, 0.5, "No Per-Class Metrics Found", ha='center', fontsize=12)

    plt.tight_layout()
    print("Showing evaluation metrics plot...")
    plt.show()
    
def create_safe_stratify_keys(y_np, min_samples=3):
    """
    Creates stratification keys where any label combination with fewer than 
    min_samples is grouped into a 'rare_pattern' bucket.
    """
    from collections import Counter
    
    y_str = ["".join(map(str, row.astype(int))) for row in y_np]
    
    key_counts = Counter(y_str)
    
    safe_keys = []
    for key in y_str:
        if key_counts[key] >= min_samples:
            safe_keys.append(key)
        else:
            safe_keys.append("rare_pattern")
            
    rare_count = safe_keys.count("rare_pattern")
    if 0 < rare_count < min_samples:
        print(f"Warning: 'rare_pattern' count ({rare_count}) is too low. Merging into majority.")
        most_common = key_counts.most_common(1)[0][0]
        safe_keys = [k if k != "rare_pattern" else most_common for k in safe_keys]
        
    return safe_keys

if __name__ == "__main__":
    
    raw_df = folder_to_dataframe('../code_classification_dataset')
    clean_df = clean_format_dataframe(raw_df)
    y = get_labels(clean_df) 
    X, sorted_documents = load_embeddings_from_chroma(db_path="./my_vector_db", collection_name="code_problems")
    
    y_np = np.array(y)
    
    stratify_keys = create_safe_stratify_keys(y_np, min_samples=3)

    print("Attempting to split data with Stratification...")

    try:
        X_train, X_temp, y_train, y_temp, _, stratify_temp = train_test_split(
            X, y_np, stratify_keys, 
            test_size=0.25, 
            random_state=42, 
            stratify=stratify_keys 
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, 
            random_state=42, 
            stratify=stratify_temp 
        )
        print("Success! Data stratified correctly.")

    except ValueError as e:
        print(f"\n!!! Stratification Failed: {e}")
        print("Falling back to RANDOM split (distributions may be slightly imbalanced).")
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_np, test_size=0.25, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    print(f"Train size: {len(X_train)} (75%)")
    print(f"Val size:   {len(X_val)} (12.5%)")
    print(f"Test size:  {len(X_test)} (12.5%)")
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    best_optuna_params = run_optuna_optimization()

    final_lr = best_optuna_params.get('lr', 0.001)
    final_batch_size = best_optuna_params.get('batch_size', 32)
    final_hidden_size = best_optuna_params.get('hidden_size', None)
    final_epochs = best_optuna_params.get('epochs', 20)

    print(f"Final Config: LR={final_lr}, Batch={final_batch_size}, Hidden={final_hidden_size}, Epochs={final_epochs}")

    trained_model, train_history, val_history = train_model(
        X_train, y_train, 
        X_val, y_val, 
        input_size=768, 
        num_classes=8,
        epochs=final_epochs, 
        batch_size=final_batch_size, 
        lr=final_lr, 
        hidden_size=final_hidden_size
    )

    evaluate_model_loss(trained_model, X_test, y_test)
    test_metrics_data = evaluate_model_metrics(trained_model, X_test, y_test, class_names=['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities'])
    
    plot_training_history(train_history, val_history)
    plot_test_metrics(test_metrics_data)
    save_dir = './NN_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    save_path = os.path.join(save_dir, 'final_model.pth')

    torch.save(trained_model.state_dict(), save_path)
    
    print(f"Model successfully saved to: {save_path}")
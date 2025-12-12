import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, precision_recall_fscore_support
from sklearn.preprocessing import normalize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.transform_data import clean_format_dataframe, folder_to_dataframe, load_embeddings_from_chroma, get_labels

class MultiLabelCentroidClassifier:
    def __init__(self):
        self.centroids = None
        self.best_thresholds = None 

    def fit(self, X_train, y_train):
        """
        Calculates the centroid for each class.
        """
        X = np.array(X_train)
        y = np.array(y_train)
        
        n_classes = y.shape[1]
        embed_dim = X.shape[1]
        
        self.centroids = np.zeros((n_classes, embed_dim))
        
        print(f"Fitting centroids for {n_classes} classes...")
        
        for i in range(n_classes):
            relevant_indices = np.where(y[:, i] == 1)[0]
            
            if len(relevant_indices) > 0:
                class_vectors = X[relevant_indices]
                centroid = np.mean(class_vectors, axis=0)
                self.centroids[i] = centroid / np.linalg.norm(centroid)
            else:
                print(f"Warning: Class {i} has no samples.")
        
        return self

    def predict_proba(self, X):
        X = np.array(X)
        X_norm = normalize(X, axis=1, norm='l2')
        similarities = np.dot(X_norm, self.centroids.T)
        return similarities

    def tune_threshold(self, X_train, y_train):
        """
        Automatically finds the best threshold FOR EACH CLASS individually.
        """
        print("Tuning per-class thresholds on validation set...")
        scores = self.predict_proba(X_train)
        y_true = np.array(y_train)
        
        n_classes = y_true.shape[1]
        self.best_thresholds = np.full(n_classes, 0.5)
        
        for i in range(n_classes):
            y_true_col = y_true[:, i]
            scores_col = scores[:, i]
            
            best_f1 = -1
            best_t = 0.5
            
            for t in np.arange(0.1, 0.96, 0.05):
                y_pred_col = (scores_col > t).astype(int)
                f1 = f1_score(y_true_col, y_pred_col, average='binary', zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            
            self.best_thresholds[i] = best_t
            
        print(f"Tuning Complete. Avg Threshold: {np.mean(self.best_thresholds):.2f}")
        return self.best_thresholds

    def predict(self, X):
        scores = self.predict_proba(X)
        if self.best_thresholds is None:
            return (scores > 0.5).astype(int)
        return (scores > self.best_thresholds).astype(int)


def plot_test_metrics(metrics_data):
    """
    Plots the final test metrics:
    1. Bar chart for Global Metrics.
    2. Table for Per-Class Metrics.
    """
    global_data = {}
    class_data_raw = {}
    
    std_aggregates = ["Global_F1_Micro", "Global_F1_Macro", "Global_Precision_Micro", 
                      "Global_Recall_Micro", "Global_Hamming_Loss"]

    for k, v in metrics_data.items():
        if k in std_aggregates:
            global_data[k] = v
        else:
            class_data_raw[k] = v

    parsed_data = {}
    suffixes = ["_F1", "_Precision", "_Recall"]
    
    for k, v in class_data_raw.items():
        for suffix in suffixes:
            if k.endswith(suffix):
                class_name = k[:-len(suffix)]  
                metric_name = suffix.strip('_') 
                
                if class_name not in parsed_data:
                    parsed_data[class_name] = {}
                parsed_data[class_name][metric_name] = v
                break
    
    if parsed_data:
        df_class = pd.DataFrame.from_dict(parsed_data, orient='index')
        cols = [c for c in ["F1", "Precision", "Recall"] if c in df_class.columns]
        df_class = df_class[cols]
        if "F1" in df_class.columns:
            df_class = df_class.sort_values(by="F1", ascending=False)
    else:
        df_class = pd.DataFrame()

    fig_height = 8 + (len(df_class) * 0.3) if not df_class.empty else 6
    fig, axes = plt.subplots(2, 1, figsize=(12, fig_height), gridspec_kw={'height_ratios': [0.8, 1.2]})

    if global_data:
        names = list(global_data.keys())
        values = list(global_data.values())
        clean_names = [n.replace("Global_", "").replace("_", " ") for n in names]
        
        sns.barplot(x=clean_names, y=values, palette="viridis", ax=axes[0], hue=clean_names, legend=False)
        axes[0].set_title('Global Evaluation Metrics', fontsize=16, fontweight='bold')
        axes[0].set_ylim(0, 1.1)
        
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
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')

        axes[1].set_title('Per-Class Metrics Breakdown', fontsize=16, pad=20, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, "No Per-Class Metrics Found", ha='center', fontsize=12)

    plt.tight_layout()
    print("Showing evaluation metrics plot...")
    plt.show()

def evaluate_and_plot(model, X_test, y_test, class_names):
    """
    Computes metrics and triggers the plotting function.
    """
    y_pred = model.predict(X_test)
    y_true = np.array(y_test)
    
    metrics_data = {
        "Global_F1_Micro": f1_score(y_true, y_pred, average='micro', zero_division=0),
        "Global_F1_Macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "Global_Precision_Micro": precision_score(y_true, y_pred, average='micro', zero_division=0),
        "Global_Recall_Micro": recall_score(y_true, y_pred, average='micro', zero_division=0),
        "Global_Hamming_Loss": hamming_loss(y_true, y_pred)
    }

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    for i, name in enumerate(class_names):
        metrics_data[f"{name}_F1"] = f1[i]
        metrics_data[f"{name}_Precision"] = precision[i]
        metrics_data[f"{name}_Recall"] = recall[i]

    plot_test_metrics(metrics_data)

if __name__ == "__main__":
    raw_df = folder_to_dataframe('../code_classification_dataset')
    clean_df = clean_format_dataframe(raw_df)
    y = get_labels(clean_df)
    class_names = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
    X, sorted_documents = load_embeddings_from_chroma(db_path="./my_vector_db", collection_name="code_problems")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.125, random_state=42
    )

    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print(f"Classes:    {class_names}")

    clf = MultiLabelCentroidClassifier()
    clf.fit(X_train, y_train)

    thresholds = clf.tune_threshold(X_train, y_train)

    print(f"Using thresholds: {thresholds}")
    evaluate_and_plot(clf, X_test, y_test, class_names)
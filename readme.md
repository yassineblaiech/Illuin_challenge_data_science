# Algorithm Exercise Classification Challenge 

## 1. Context & Objective
This project addresses the **Tech Data Science Challenge** to classify algorithmic exercises from Codeforces into specific categories (tags). 

The goal is to develop a multi-label classification system that predicts the relevant tags for a given programming problem description. The solution focuses specifically on the following 8 target tags:
`['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']`.

## 2. Methodology

The approach is divided into distinct stages: Data Preparation, Vector Embedding, and two modeling strategies (Geometric Baseline vs. Deep Learning).

### Step 1: Data Ingestion & Cleaning
**File:** `utils/transform_data.py`

The raw dataset consists of individual JSON files. A pipeline was implemented to:
1.  **Ingest:** Iterate through the dataset folder and merge all valid JSON files into a single Pandas DataFrame.
2.  **Filter:** Reduce the scope to the 8 target tags mentioned above.
3.  **Format:** One-hot encode the tags (creating binary columns like `tag_math`, `tag_graphs`) to prepare for multi-label classification.
4.  **Clean:** Handle missing values in critical text fields (`input_spec`, `output_spec`, `description`, `notes`).

### Step 2: Semantic Embedding (Vector Database)
**File:** `utils/embedding_function.py` & `encode_data`

Instead of traditional TF-IDF, this solution utilizes a **Large Language Model (Gemma)** to generate semantic embeddings.
* **Text Aggregation:** For each problem, the `input_spec`, `output_spec`, `description`, and `notes` are concatenated into a single text block.
* **Vectorization:** Each text block is passed through the Gemma embedding model to produce a high-dimensional vector (768 dimensions).
* **Storage:** Vectors are persisted in a **ChromaDB** database for efficient retrieval.

### Step 3: Modeling Approaches
I implemented two different approaches to solve the classification problem.

#### Step 3A: Geometric Baseline (Centroid Classifier)
**File:** `models/centroid_classifier.py`
A lightweight, interpretable model based on cosine similarity.

* **Training:** The model computes the "center of mass" (mean vector) for all training examples belonging to a specific tag (e.g., the average "Graph" vector).
* **Inference:** It calculates the cosine similarity between a new problem's vector and the 8 class centroids.
* **Adaptive Thresholding:** Crucially, it does not use a fixed cutoff. It iterates through the validation set to find the optimal similarity threshold *per class* (e.g., "Geometry" might need a 0.85 similarity score, while "Math" needs 0.60).

#### Step 3B: Deep Learning Approach (Neural Network)
**File:** `models/neural_network.py`
A PyTorch-based Multi-Layer Perceptron (MLP) designed to capture non-linear relationships.

* **Architecture:** A fully connected network (Input → Hidden Layer + ReLU → Output + Sigmoid).
* **Optimization:** Uses Binary Cross Entropy (BCE) Loss, suitable for multi-label tasks, and the Adam optimizer.
* **Advanced Features:**
    * **Stratified Splitting:** Implements custom logic to handle data imbalances and ensure rare tag combinations are present in train/val/test sets.
    * **Hyperparameter Tuning:** Utilizes **Optuna** to automatically search for the best learning rate, batch size, and hidden layer size.
    * **Early Stopping:** Monitors validation loss to prevent overfitting.

#### Step 3C: Experimental Approach (LLM Fine-Tuning)
**Status:** *Experimental / Research*
I attempted to directly fine-tune a Generative LLM (`Qwen/Qwen2.5-Coder-7B` and `Qwen/Qwen3-1.7B`) to output tags as text.
* **Technique:** Supervised Fine-Tuning (SFT) using **QLoRA** (4-bit quantization + Low-Rank Adapters) to run on consumer GPUs (Colab T4).
* **Prompting:** Formatted inputs as instructions: *"Given the following problem description, classify it..."*.
* **Outcome:** While promising, this approach was **computationally expensive** and slower at inference time (seconds vs. milliseconds). It was difficult to stabilize the text output for strict multi-label evaluation compared to the vector-based approaches.

## 3. Evaluation & Visualization

Both models are evaluated on a hold-out test set (12.5% of data). The primary metrics are **F1 Score (Micro & Macro)** and **Hamming Loss**.

To interpret results, a unified plotting module generates:
1.  **Global Metrics Chart:** A visual summary of overall performance.
2.  **Per-Class Breakdown:** A detailed table showing Precision, Recall, and F1 for every individual tag to identify specific strengths and weaknesses (e.g., distinguishing `math` from `number theory`).

## 4. How to Run

### Prerequisites
* Python 3.8+
* Dependencies: `torch`, `optuna`, `pandas`, `numpy`, `scikit-learn`, `chromadb`, `matplotlib`, `seaborn`, `tqdm`.

### Execution

**Option A: Run the Centroid Classifier (Fast & Interpretable)**
```bash
python classification_with_embeddings\centroid_approach.py

**Option B: Run the Neural Network (High Capacity & Optimized)**
```bash
python classification_with_embeddings\neural_network_for_classification.py

**Option C: Run the fine tuning approach (a tested method that didn't show very promising results)**
supervised_fine_tuning\llm_fine_tuning_for_classification.ipynb





import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from searchless_chess.src import tokenizer
from searchless_chess.src.engines import constants as engine_constants

# 1. Function to Collect Hidden States and Labels
def collect_hidden_states(engine, dataset_file, max_samples=500):
    """
    Collect hidden states from the transformer model for chess positions and their labels.
    
    Args:
        dataset_file (str): Path to the dataset file with FEN strings and labels.
        max_samples (int): Maximum number of samples to process.
    
    Returns:
        np.ndarray: Hidden states array of shape (n_samples, n_features).
        np.ndarray: Labels array of shape (n_samples,).
    """
    hidden_states = []
    labels = []
    with open(dataset_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            fen, label = line.strip().split("\t")
            label = int(label)
            tokens = tokenizer.tokenize(fen).astype(np.int32)
            dummy_action = np.array([0], dtype=np.int32)
            dummy_r = np.array([0], dtype=np.int32)
            sequence = np.concatenate([tokens, dummy_action, dummy_r])[None, :]
            _, _, h = engine.predict_fn(sequence)
            hidden_state = h[0, 78, :]  # Extract hidden state at index 78
            hidden_states.append(hidden_state)
            labels.append(label)
    return np.array(hidden_states), np.array(labels)

# 2. Aggressive Quantile-Based Clipping
def aggressive_quantile_clip(X, lower_quantile=0.05, upper_quantile=0.95):
    """
    Clip hidden states based on quantiles to dynamically handle extreme values.
    
    Args:
        X (np.ndarray): Hidden states.
        lower_quantile (float): Lower quantile for clipping.
        upper_quantile (float): Upper quantile for clipping.
    
    Returns:
        np.ndarray: Clipped hidden states.
    """
    lower_bound = np.quantile(X, lower_quantile, axis=0)
    upper_bound = np.quantile(X, upper_quantile, axis=0)
    return np.clip(X, lower_bound, upper_bound)

# 3. Logarithmic Transformation
def log_transform(X):
    """
    Apply logarithmic transformation to compress large values while preserving sign.
    
    Args:
        X (np.ndarray): Hidden states.
    
    Returns:
        np.ndarray: Transformed hidden states.
    """
    return np.log1p(np.abs(X)) * np.sign(X)

# 4. Main Probing Function
def probe_concept(hidden_states, labels, output_file=None):
    """
    Probe the hidden states to detect a concept using a random forest classifier.
    
    Args:
        hidden_states (np.ndarray): Hidden states from the model.
        labels (np.ndarray): Binary labels indicating the presence of the concept.
        output_file (file): File object to write results to.
    """

    # Replace NaNs and infinite values
    hidden_states = np.nan_to_num(hidden_states, nan=0.0, posinf=1e5, neginf=-1e5)

    # Define the pipeline
    pipeline = Pipeline([
        ('clip', FunctionTransformer(aggressive_quantile_clip)),  # Clip outliers
        ('log', FunctionTransformer(log_transform)),  # Compress large values
        ('variance', VarianceThreshold(threshold=1e-4)),  # Remove near-zero variance features
        ('scaler', MinMaxScaler()),  # Scale to [0, 1]
        ('feature_select', SelectKBest(f_classif, k=100)),  # Select top 100 features
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42))  # Regularized classifier
    ])

    # Perform 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1']
    results = cross_validate(pipeline, hidden_states, labels, cv=cv, scoring=scoring, return_train_score=True)

    # Report results
    result_str = "Cross-validated Results:\n"
    result_str += f"Mean Train Accuracy: {np.mean(results['train_accuracy']):.3f}\n"
    result_str += f"Mean Test Accuracy: {np.mean(results['test_accuracy']):.3f}\n"
    result_str += f"Standard Deviation of Test Accuracy: {np.std(results['test_accuracy']):.3f}\n"
    result_str += f"Mean Test F1 Score: {np.mean(results['test_f1']):.3f}\n"
    result_str += f"Standard Deviation of Test F1 Score: {np.std(results['test_f1']):.3f}\n\n"
    
    if output_file:
        output_file.write(result_str)
    else:
        print(result_str)

# 5. Run the Probing Task
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load the model
    ENGINE_9M = engine_constants.ENGINE_BUILDERS['9M']()
    ENGINE_270M = engine_constants.ENGINE_BUILDERS['270M']()

    MAX_SAMPLES = 1000

    # Open a file for writing results
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace('.', '-')
    results_file_path = os.path.join(results_dir, f"probe_results_{timestamp}.txt")
    
    with open(results_file_path, "w") as results_file:
        # Collect hidden states and labels
        for label_file in ['in_check.txt', 'bishop_pair.txt', 'high_mobility.txt', 'material_difference.txt', 'passed_pawn.txt', 'pinned_piece.txt']:
            hidden_states_9, labels_9 = collect_hidden_states(ENGINE_9M, f"label/{label_file}", max_samples=MAX_SAMPLES)
            hidden_states_270, labels_270 = collect_hidden_states(ENGINE_270M, f"label/{label_file}", max_samples=MAX_SAMPLES)

            results_file.write(f"Probing {label_file} with 9M:\n")
            probe_concept(hidden_states_9, labels_9, results_file)

            results_file.write(f"Probing {label_file} with 270M:\n")
            probe_concept(hidden_states_270, labels_270, results_file)

            print(f"Probing {label_file} with 9M and 270M complete")
    
    print(f"Results written to {results_file_path}")

'''
Before running:
cd searchless_chess/src
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export PYTHONPATH=$(pwd)/../..
'''
import numpy as np
import os
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from searchless_chess.src import tokenizer, utils
from searchless_chess.src.engines import constants as engine_constants
import chess

# 1. Function to collect attention weights and hidden states
def collect_data(engine, dataset_file, max_samples=500):
    """
    Collect attention weights, Euclidean distances, and hidden states for chess positions and their labels.
    
    Args:
        engine: Chess engine with transformer model.
        dataset_file (str): Path to the dataset file with FEN strings and labels.
        max_samples (int): Maximum number of samples to process.
    
    Returns:
        np.ndarray: Attention-weighted distance features array of shape (n_samples, 1).
        np.ndarray: Hidden states array of shape (n_samples, n_features).
        np.ndarray: Labels array of shape (n_samples,).
    """
    attention_distances = []
    hidden_states = []
    labels = []
    
    with open(dataset_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
                
            fen, label = line.strip().split("\t")
            label = int(label)
            
            # Create a chess board from FEN
            board = chess.Board(fen)
            
            # Get engine move
            move = engine.play(board)
            
            # Process for attention distances
            # 1) Tokenize board position
            tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
            action = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
            dummy_r = np.array([0], dtype=np.int32)
            
            # Combine sequence
            sequence = np.concatenate([tokenized_fen, action, dummy_r])[None, :]  # shape [1,79]
            
            # 2) Run inference
            _, attention_weights, h = engine.predict_fn(sequence)
            
            # 3) Extract last-token-to-board-squares attention
            last_token_attention = [
                layer[0, :, 78, 1:65]  # shape = [num_heads, 64]
                for layer in attention_weights
            ]
            
            # 4) Average over heads/layers
            avg_attention_per_layer = [np.mean(heads, axis=0) for heads in last_token_attention]
            avg_attention = np.mean(avg_attention_per_layer, axis=0)  # shape [64]
            
            # 5) Normalize attention weights
            attn_min, attn_max = avg_attention.min(), avg_attention.max()
            if attn_max > attn_min:
                attn_normalized = (avg_attention - attn_min) / (attn_max - attn_min)
            else:
                attn_normalized = np.zeros_like(avg_attention)
            
            # 6) Calculate attention-weighted Euclidean distance
            distance = calculate_attention_weighted_distance(move, attn_normalized)
            attention_distances.append(distance)
            
            # Store hidden state for linear probing
            hidden_state = h[0, 78, :]  # Extract hidden state at index 78
            hidden_states.append(hidden_state)
            
            # Store label
            labels.append(label)
            
    return (np.array(attention_distances).reshape(-1, 1), 
            np.array(hidden_states), 
            np.array(labels))

# 2. Function to calculate attention-weighted Euclidean distance
def calculate_attention_weighted_distance(move, attention_weights):
    """
    Calculate attention-weighted Euclidean distance from the move to each square.
    
    Args:
        move (chess.Move): The chess move being made.
        attention_weights (np.ndarray): Normalized attention weights for each square.
    
    Returns:
        float: Weighted average Euclidean distance.
    """
    # Get source and target square coordinates
    from_file, from_rank = chess.square_file(move.from_square), chess.square_rank(move.from_square)
    to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
    
    # For each board square, calculate Euclidean distance to both source and target
    weighted_distances = 0
    total_weight = np.sum(attention_weights)
    
    if total_weight == 0:
        return 0  # To avoid division by zero
    
    for sq in range(64):
        file, rank = chess.square_file(sq), chess.square_rank(sq)
        
        # Euclidean distance to source square
        from_dist = np.sqrt((file - from_file)**2 + (rank - from_rank)**2)
        
        # Euclidean distance to target square
        to_dist = np.sqrt((file - to_file)**2 + (rank - to_rank)**2)
        
        # Average distance to source and target
        avg_dist = (from_dist + to_dist) / 2
        
        # Weight by attention
        weight = attention_weights[sq]
        weighted_distances += avg_dist * weight
    
    # Return weighted average distance
    return weighted_distances / total_weight

# 3. Function to probe the attention distance feature
def probe_attention_distance(features, labels, output_file=None):
    """
    Probe the attention-distance features to detect a concept using a random forest classifier.
    
    Args:
        features (np.ndarray): Attention-distance features.
        labels (np.ndarray): Binary labels indicating the presence of the concept.
        output_file (file): File object to write results to.
    """
    # Simple classifier as features are just 1D
    classifier = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    
    # Perform 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'f1']
    results = cross_validate(classifier, features, labels, cv=cv, scoring=scoring, return_train_score=True)
    
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

# Function to calculate and print average attention distance
def print_average_attention_distance(attention_distances, model_name, label_file, output_file=None):
    """
    Calculate and print the average attention distance.
    
    Args:
        attention_distances (np.ndarray): Attention-distance features.
        model_name (str): The name of the model.
        label_file (str): The label file being processed.
        output_file (file): File object to write results to.
    """
    avg_distance = np.mean(attention_distances)
    std_distance = np.std(attention_distances)
    result_str = f"Average attention distance for {model_name} on {label_file}: {avg_distance:.4f}\n"
    result_str += f"Standard deviation of attention distance: {std_distance:.4f}\n\n"
    
    if output_file:
        output_file.write(result_str)
    else:
        print(result_str)

# 4. Functions from linear_probe.py
def aggressive_quantile_clip(X, lower_quantile=0.05, upper_quantile=0.95):
    """
    Clip hidden states based on quantiles to dynamically handle extreme values.
    """
    lower_bound = np.quantile(X, lower_quantile, axis=0)
    upper_bound = np.quantile(X, upper_quantile, axis=0)
    return np.clip(X, lower_bound, upper_bound)

def log_transform(X):
    """
    Apply logarithmic transformation to compress large values while preserving sign.
    """
    return np.log1p(np.abs(X)) * np.sign(X)

def probe_hidden_states(hidden_states, labels, output_file=None):
    """
    Probe the hidden states to detect a concept using a random forest classifier.
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

# 5. Run the Combined Experiment
if __name__ == "__main__":
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load the models
    ENGINE_9M = engine_constants.ENGINE_BUILDERS['9M']()
    ENGINE_270M = engine_constants.ENGINE_BUILDERS['270M']()

    MAX_SAMPLES = 10

    # Open a file for writing results
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace('.', '-')
    results_file_path = os.path.join(results_dir, f"results_{timestamp}.txt")
    
    with open(results_file_path, "w") as results_file:
        # Collect data and run probing for each concept
        for label_file in ['in_check.txt', 'bishop_pair.txt', 'high_mobility.txt', 'material_difference.txt', 'passed_pawn.txt', 'pinned_piece.txt']:
            # Process with 9M model
            attn_dist_9m, hidden_states_9m, labels_9m = collect_data(ENGINE_9M, f"label/{label_file}", max_samples=MAX_SAMPLES)
            
            results_file.write(f"Processing {label_file} with 9M (Attention Distance):\n")
            print_average_attention_distance(attn_dist_9m, "9M", label_file, results_file)
            
            results_file.write(f"Probing {label_file} with 9M (Hidden States):\n")
            probe_hidden_states(hidden_states_9m, labels_9m, results_file)
            
            # Process with 270M model
            attn_dist_270m, hidden_states_270m, labels_270m = collect_data(ENGINE_270M, f"label/{label_file}", max_samples=MAX_SAMPLES)
            
            results_file.write(f"Processing {label_file} with 270M (Attention Distance):\n")
            print_average_attention_distance(attn_dist_270m, "270M", label_file, results_file)
            
            results_file.write(f"Probing {label_file} with 270M (Hidden States):\n")
            probe_hidden_states(hidden_states_270m, labels_270m, results_file)
            
            print(f"Combined processing of {label_file} with 9M and 270M complete")
    
    print(f"Results written to {results_file_path}")

'''
Before running:
cd searchless_chess/src
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export PYTHONPATH=$(pwd)/../..
''' 
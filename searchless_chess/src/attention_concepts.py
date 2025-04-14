import chess
import numpy as np
from searchless_chess.src import tokenizer, utils
from searchless_chess.src.engines import constants as engine_constants

# Load the 270M-parameter chess model
engine = engine_constants.ENGINE_BUILDERS['9M']()

# Compute quantitative metrics for attention with vs. without concept
def analyze_concept_metrics(dataset_file="label/in_check.txt"):
    # Lists to store attention weights for all squares
    concept_attns = []  # Label 1 (concept present)
    no_concept_attns = []  # Label 0 (concept absent)

    # Read all positions from the dataset
    with open(dataset_file, "r") as f:
        lines = 0
        for line in f:
            if lines > 100:
                break
            lines += 1
            fen, label = line.strip().split("\t")
            label = int(label)
            board = chess.Board(fen)
            move = engine.play(board)
            tokens = tokenizer.tokenize(fen).astype(np.int32)
            action = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
            sequence = np.concatenate([tokens, action, np.array([0], dtype=np.int32)])[None, :]
            _, attn_weights, _ = engine.predict_fn(sequence)
            # Get attention from last token to all 64 board squares
            square_attn = [layer[0, :, 78, 1:65] for layer in attn_weights]  # [layers, heads, 64]
            square_attn = np.array(square_attn)  # Convert to numpy for easier handling
            if label == 1:
                concept_attns.append(square_attn)
            else:
                no_concept_attns.append(square_attn)

    print(f"Analyzing {len(concept_attns)} concept-present and {len(no_concept_attns)} concept-absent positions.")

    # Convert to numpy arrays: [n_positions, layers, heads, 64]
    concept_attns = np.array(concept_attns)
    no_concept_attns = np.array(no_concept_attns)

    # Compute metrics
    num_layers, num_heads = concept_attns.shape[1], concept_attns.shape[2]

    # 1. Total Attention Magnitude
    concept_magnitude = np.mean(np.sum(concept_attns, axis=-1))  # Sum over squares, mean over pos/layers/heads
    no_concept_magnitude = np.mean(np.sum(no_concept_attns, axis=-1))

    # 2. Attention Entropy
    def compute_entropy(attn):
        # Normalize to sum to 1 per head
        attn_sum = np.sum(attn, axis=-1, keepdims=True)
        attn_sum = np.where(attn_sum == 0, 1e-10, attn_sum)  # Avoid division by zero
        attn_norm = attn / attn_sum
        # Compute entropy: -sum(p * log(p)), handle zeros
        entropy = -np.sum(attn_norm * np.log2(attn_norm + 1e-10), axis=-1)
        return np.mean(entropy)  # Mean over pos/layers/heads

    concept_entropy = compute_entropy(concept_attns)
    no_concept_entropy = compute_entropy(no_concept_attns)

    # 3. Maximum Attention Weight
    concept_max = np.mean(np.max(concept_attns, axis=-1))
    no_concept_max = np.mean(np.max(no_concept_attns, axis=-1))

    # 4. KL Divergence (average attention distributions)
    concept_avg_dist = np.mean(concept_attns, axis=0).mean(axis=0).mean(axis=0)  # [64]
    no_concept_avg_dist = np.mean(no_concept_attns, axis=0).mean(axis=0).mean(axis=0)  # [64]
    # Normalize distributions
    concept_avg_dist = concept_avg_dist / (np.sum(concept_avg_dist) + 1e-10)
    no_concept_avg_dist = no_concept_avg_dist / (np.sum(no_concept_avg_dist) + 1e-10)
    # Compute KL divergence
    kl_div = np.sum(concept_avg_dist * np.log2(concept_avg_dist / (no_concept_avg_dist + 1e-10) + 1e-10))

    # Output results
    concept_name = dataset_file.split("/")[-1].replace(".txt", "")
    print(f"\n📊 Quantitative Metrics for '{concept_name}' 📊")
    print(f"1. Total Attention Magnitude:")
    print(f"   - With concept: {concept_magnitude:.3f}")
    print(f"   - Without concept: {no_concept_magnitude:.3f}")
    print(f"   - Difference: {concept_magnitude - no_concept_magnitude:.3f}")
    print(f"2. Attention Entropy (bits):")
    print(f"   - With concept: {concept_entropy:.3f}")
    print(f"   - Without concept: {no_concept_entropy:.3f}")
    print(f"   - Difference: {concept_entropy - no_concept_entropy:.3f}")
    print(f"3. Maximum Attention Weight:")
    print(f"   - With concept: {concept_max:.3f}")
    print(f"   - Without concept: {no_concept_max:.3f}")
    print(f"   - Difference: {concept_max - no_concept_max:.3f}")
    print(f"4. KL Divergence (bits): {kl_div:.3f}")

if __name__ == "__main__":
    analyze_concept_metrics("label/in_check.txt")
    analyze_concept_metrics("label/pinned_piece.txt")
    analyze_concept_metrics("label/passed_pawn.txt")
    analyze_concept_metrics("label/bishop_pair.txt")
    analyze_concept_metrics("label/material_difference.txt")
    analyze_concept_metrics("label/high_mobility.txt")
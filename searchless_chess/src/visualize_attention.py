import chess
import chess.svg
import numpy as np
from searchless_chess.src import tokenizer, utils
from searchless_chess.src.engines import constants as engine_constants

def visualize_attention(board, move, output_file="attention_heatmap.svg"):
    """
    Visualizes the attention weights of the transformer model on the chessboard
    for a given move, ensuring that the squares are highlighted correctly.
    """
    engine = engine_constants.ENGINE_BUILDERS['270M']()

    # 1) Tokenize the board FEN plus the chosen move and the dummy return bucket.
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    action = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
    dummy_r = np.array([0], dtype=np.int32)  # Dummy return token

    # tokenized_fen has length 77: index 0 is side-to-move, 1..64 = squares,
    # 65..76 = castling/en-passant/halfmove/fullmove
    # Then we add 1 token for the action and 1 token for the dummy return => total 79.
    sequence = np.concatenate([tokenized_fen, action, dummy_r])[None, :]  # shape [1,79]

    logits, attention_weights, hidden_states = engine.predict_fn(sequence)
    # attention_weights is a list of shape [B, num_heads, T, T] for each layer
    # so each layer[i]: [1, num_heads, 79, 79]

    # 2) We want the attention from the *last token* (index=78) to the
    #    *64 board squares*, which in the FEN tokenizer are at indices 1..64.
    # Hence we slice columns [1:65], *not* [:64].
    last_token_attention = [
        layer[0, :, 78, 1:65]  # shape = [num_heads, 64]
        for layer in attention_weights
    ]

    # 3) Average over all heads and then average over all layers
    #    so we get one array of length 64 (one attention weight per square).
    avg_attention_per_layer = [np.mean(heads, axis=0) for heads in last_token_attention]  # each shape [64]
    avg_attention = np.mean(avg_attention_per_layer, axis=0)  # shape [64]

    # Normalize the attention so we can map to an opacity 0..1
    attn_min, attn_max = avg_attention.min(), avg_attention.max()
    if attn_max > attn_min:  # avoid division-by-zero
        attn_normalized = (avg_attention - attn_min) / (attn_max - attn_min)
    else:
        attn_normalized = np.zeros_like(avg_attention)

    # 4) Convert from FEN-index (top-rank-first) to python-chess indexing.
    #    In the tokenizer, i=0 => a8, i=7 => h8, ..., i=63 => h1.
    #    python-chess says a1=0, b1=1, ... h8=63.  So we do:
    def fen_token_index_to_square(i):
        row = i // 8  # 0..7, top-rank=0
        col = i % 8
        return 8 * (7 - row) + col  # re-map so row=0 => rank8 => squares 56..63

    # Color function
    def weight_to_color(weight, max_opacity=0.7):
        return f"rgba(255, 0, 0, {weight * max_opacity})"

    square_colors = {}
    for i, weight in enumerate(attn_normalized):
        if weight > 0.1:  # only highlight squares with "significant" attention
            real_sq = fen_token_index_to_square(i)
            square_colors[real_sq] = weight_to_color(weight)

    # Generate and save the SVG with chess.svg
    svg = chess.svg.board(
        board=board,
        fill=square_colors,  # dictionary keyed by python-chess square (0..63)
        size=800,
        lastmove=move,
    )
    with open(output_file, "w") as f:
        f.write(svg)

    print(f"Attention heatmap for move {move} saved to {output_file}")


if __name__ == "__main__":
    board = chess.Board()

    engine = engine_constants.ENGINE_BUILDERS['270M']()
    best_move = engine.play(board)
    visualize_attention(board, best_move)

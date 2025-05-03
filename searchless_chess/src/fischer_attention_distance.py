'''
compares how focused the attention is in two transformer-based chess models
(9M and 270M) by computing the attention-weighted distance from the source square
of the last move played on a board

It runs each model on a given chess position,
extracts the attention each model places on all 64 squares, and weights those values
by how far each square is from the move’s origin. A lower score suggests the model is
paying closer attention to nearby squares—typically more human-like reasoning—while a
higher score indicates more distributed, abstract focus.
'''
import numpy as np
import chess
from searchless_chess.src import tokenizer, utils
from searchless_chess.src.engines import constants as engine_constants

def square_to_coords(square):
    return divmod(square, 8)

def manhattan_dist(sq1, sq2):
    r1, c1 = square_to_coords(sq1)
    r2, c2 = square_to_coords(sq2)
    return abs(r1 - r2) + abs(c1 - c2)

def weighted_distance_score(attn_weights, ref_square):
    return sum(manhattan_dist(i, ref_square) * w for i, w in enumerate(attn_weights))

def get_attention_weights(board, move, model_name="9M"):
    engine = engine_constants.ENGINE_BUILDERS[model_name]()
    tokenized = tokenizer.tokenize(board.fen()).astype(np.int32)
    action = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
    dummy_r = np.array([0], dtype=np.int32)
    sequence = np.concatenate([tokenized, action, dummy_r])[None, :]
    _, attn_weights, _ = engine.predict_fn(sequence)

    per_layer = [np.mean(layer[0, :, 78, 1:65], axis=0) for layer in attn_weights]
    return np.mean(per_layer, axis=0)

def analyze_board(board):
    results = {}
    for model_name in ["9M", "270M"]:
        engine = engine_constants.ENGINE_BUILDERS[model_name]()
        move = engine.play(board)
        attn = get_attention_weights(board, move, model_name)
        score = weighted_distance_score(attn, move.from_square)
        results[model_name] = (score, move.uci())
    return results

if __name__ == "__main__":
    board = chess.Board()
    result = analyze_board(board)
    print("Weighted attention distances from last move:")
    for model, (score, move) in result.items():
        print(f"{model}: {score:.2f} (move: {move})")

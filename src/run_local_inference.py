import chess

from searchless_chess.src.engines import constants as engine_constants

def main():
    engine = engine_constants.ENGINE_BUILDERS['270M']()
    board = chess.Board()
    print("Initial board FEN:", board.fen())

    best_move = engine.play(board)
    print("Model's recommended move:", best_move.uci())

if __name__ == "__main__":
    main()


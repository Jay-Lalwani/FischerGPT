import chess
import chess.pgn

# Define piece values
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def material_difference(board):
    """Calculate material difference (White - Black)."""
    white_material = 0
    black_material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    return abs(white_material - black_material)

# Track unique FENs
seen = set()

# Initialize counters for balancing
material_counts = {0: 0, 1: 0}
check_counts = {0: 0, 1: 0}

# Process PGN and write tagged FENs
with open("material_difference.txt", "w") as f1:
    with open("in_check.txt", "w") as f2:
        with open("lichess.pgn", "r") as pgn:
            game_count = 0
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:  # End of file
                    break
                
                # Limit to first 100 games
                game_count += 1
                if game_count > 100:
                    break
                    
                board = game.board()  # Starting position (custom if specified)
                for move in game.mainline_moves():
                    board.push(move)  # Apply move to update board
                    fen = board.fen()  # Get FEN representation
                    if fen not in seen:
                        seen.add(fen)  # Mark as seen

                        # Material Difference Labeling
                        diff = material_difference(board)
                        tag_material = 1 if diff > 3 else 0
                        # Write if the current tag is the minority or counts are equal
                        if material_counts[tag_material] <= material_counts[1 - tag_material]:
                            f1.write(f"{fen}\t{tag_material}\n")  # Write FEN and tag
                            material_counts[tag_material] += 1
                        
                        # In Check Labeling
                        tag_check = 1 if board.is_check() else 0
                        # Write if the current tag is the minority or counts are equal
                        if check_counts[tag_check] <= check_counts[1 - tag_check]:
                            f2.write(f"{fen}\t{tag_check}\n")
                            check_counts[tag_check] += 1

import chess
import chess.pgn
import random

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

def has_passed_pawn(board):
    """Check if either side has at least one passed pawn."""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type == chess.PAWN:
            # Get file and rank of the pawn
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            
            # Direction of pawn movement
            direction = 1 if piece.color == chess.WHITE else -1
            
            # Check if there are enemy pawns in front or diagonally in front
            is_passed = True
            
            # Check all ranks in front of the pawn
            for r in range(rank_idx + direction, 8 if direction > 0 else -1, direction):
                # Check the same file and adjacent files
                for f in range(max(0, file_idx - 1), min(8, file_idx + 2)):
                    check_square = chess.square(f, r)
                    blocking_piece = board.piece_at(check_square)
                    if blocking_piece is not None and blocking_piece.piece_type == chess.PAWN and blocking_piece.color != piece.color:
                        is_passed = False
                        break
                if not is_passed:
                    break
            
            if is_passed:
                return True
    return False

def has_bishop_pair(board):
    """Check if either side has a bishop pair."""
    white_bishops = 0
    black_bishops = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type == chess.BISHOP:
            if piece.color == chess.WHITE:
                white_bishops += 1
            else:
                black_bishops += 1
    return white_bishops >= 2 or black_bishops >= 2

def high_mobility(board):
    """Check if the side to move has high mobility (more than 25 legal moves)."""
    return len(list(board.legal_moves)) > 25

def has_pinned_piece(board):
    """Check if any piece is pinned on the board."""
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if board.is_pinned(piece.color, square):
                return True
    return False

def get_board_at_move_n(game, n):
    """Get the board state after the nth move."""
    board = game.board()
    moves = list(game.mainline_moves())
    if n >= len(moves):
        n = len(moves) - 1
    
    for i in range(n + 1):
        board.push(moves[i])
    
    return board

# Track unique FENs
seen = set()

# Initialize counters for balancing
material_counts = {0: 0, 1: 0}
check_counts = {0: 0, 1: 0}
passed_pawn_counts = {0: 0, 1: 0}
bishop_pair_counts = {0: 0, 1: 0}
mobility_counts = {0: 0, 1: 0}
pinned_piece_counts = {0: 0, 1: 0}

# Configuration for diversity
MAX_GAMES = 50000  # Number of games
POSITIONS_PER_GAME = 5  # Sample fewer positions per game
MIN_MOVE_NUMBER = 10  # Skip early opening moves
MOVE_PHASE_CATEGORIES = [
    (10, 20),   # Early game
    (20, 40),   # Middle game
    (40, 100)   # End game
]

# Process PGN and write tagged FENs
with open("material_difference.txt", "w") as f1:
    with open("in_check.txt", "w") as f2:
        with open("passed_pawn.txt", "w") as f3:
            with open("bishop_pair.txt", "w") as f4:
                with open("high_mobility.txt", "w") as f5:
                    with open("pinned_piece.txt", "w") as f6:
                        with open("lichess.pgn", "r") as pgn:
                            game_count = 0
                            while True:
                                game = chess.pgn.read_game(pgn)
                                if game is None:  # End of file
                                    break
                                
                                game_count += 1
                                if game_count > MAX_GAMES:
                                    break
                                
                                # Get total number of moves in the game
                                moves = list(game.mainline_moves())
                                if len(moves) < MIN_MOVE_NUMBER:
                                    continue  # Skip very short games
                                
                                # Sample positions from different phases of the game
                                sampled_move_indices = []
                                
                                # Select moves from each phase category
                                for start, end in MOVE_PHASE_CATEGORIES:
                                    if start >= len(moves):
                                        continue
                                    
                                    phase_end = min(end, len(moves))
                                    # Sample up to POSITIONS_PER_GAME/3 moves from this phase
                                    phase_samples = min(POSITIONS_PER_GAME // 3, phase_end - start)
                                    if phase_samples > 0:
                                        sampled_indices = random.sample(range(start, phase_end), phase_samples)
                                        sampled_move_indices.extend(sampled_indices)
                                
                                # If we didn't get enough samples, add more from any phase
                                if len(sampled_move_indices) < POSITIONS_PER_GAME and len(moves) > MIN_MOVE_NUMBER:
                                    remaining = POSITIONS_PER_GAME - len(sampled_move_indices)
                                    additional_indices = random.sample(
                                        range(MIN_MOVE_NUMBER, len(moves)),
                                        min(remaining, len(moves) - MIN_MOVE_NUMBER)
                                    )
                                    sampled_move_indices.extend(additional_indices)
                                
                                # Process the sampled positions
                                for move_idx in sampled_move_indices:
                                    board = get_board_at_move_n(game, move_idx)
                                    if board.is_game_over():
                                        continue  # Skip game over positions
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
                                        
                                        # Passed Pawn Labeling
                                        tag_passed_pawn = 1 if has_passed_pawn(board) else 0
                                        if passed_pawn_counts[tag_passed_pawn] <= passed_pawn_counts[1 - tag_passed_pawn]:
                                            f3.write(f"{fen}\t{tag_passed_pawn}\n")
                                            passed_pawn_counts[tag_passed_pawn] += 1
                                        
                                        # Bishop Pair Labeling
                                        tag_bishop_pair = 1 if has_bishop_pair(board) else 0
                                        if bishop_pair_counts[tag_bishop_pair] <= bishop_pair_counts[1 - tag_bishop_pair]:
                                            f4.write(f"{fen}\t{tag_bishop_pair}\n")
                                            bishop_pair_counts[tag_bishop_pair] += 1
                                        
                                        # High Mobility Labeling
                                        tag_mobility = 1 if high_mobility(board) else 0
                                        if mobility_counts[tag_mobility] <= mobility_counts[1 - tag_mobility]:
                                            f5.write(f"{fen}\t{tag_mobility}\n")
                                            mobility_counts[tag_mobility] += 1
                                        
                                        # Pinned Piece Labeling
                                        tag_pinned = 1 if has_pinned_piece(board) else 0
                                        if pinned_piece_counts[tag_pinned] <= pinned_piece_counts[1 - tag_pinned]:
                                            f6.write(f"{fen}\t{tag_pinned}\n")
                                            pinned_piece_counts[tag_pinned] += 1

print(f"Total unique positions: {len(seen)}")
print(f"Material difference positions: {sum(material_counts.values())}")
print(f"In check positions: {sum(check_counts.values())}")
print(f"Passed pawn positions: {sum(passed_pawn_counts.values())}")
print(f"Bishop pair positions: {sum(bishop_pair_counts.values())}")
print(f"High mobility positions: {sum(mobility_counts.values())}")
print(f"Pinned piece positions: {sum(pinned_piece_counts.values())}")

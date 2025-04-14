"""
Run this file (e.g. python web_chess.py), then navigate to:
http://127.0.0.1:8080/
"""

import base64
import numpy as np
from flask import (
    Flask,
    render_template_string,
    request,
    session,
    redirect,
    url_for,
    jsonify,
    flash
)

import chess
import chess.svg
import cairosvg
import os

# For AI engine
from searchless_chess.src import tokenizer, utils
from searchless_chess.src.engines import constants as engine_constants

# ----------------------------------------------------------------------
# Flask setup
# ----------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "jay"


# ----------------------------------------------------------------------
# Utilities for session-based chess
# ----------------------------------------------------------------------
def init_game_state():
    """Initialize the session state for a new game."""
    session['board_fen'] = chess.Board().fen()
    session['from_square'] = None
    session['highlighted_squares'] = {}
    session['move_history'] = []  # Initialize empty move history
    # Default model
    session['model_name'] = "9M"
    # Add player color - White by default
    session['player_color'] = chess.WHITE
    # Don't store engine in session


def load_board():
    """Return a chess.Board object from session data."""
    fen = session.get('board_fen', chess.Board().fen())
    board = chess.Board(fen)
    
    # Restore move history
    move_history = session.get('move_history', [])
    if move_history:        
        # Create a fresh board to build move history
        fresh_board = chess.Board()
        valid_moves = []
        
        # Replay each move to build the move stack
        for uci in move_history:
            try:
                move = chess.Move.from_uci(uci)
                if move in fresh_board.legal_moves:
                    fresh_board.push(move)
                    valid_moves.append(move)
                else:
                    print(f"Illegal move in history: {uci}")
            except ValueError:
                print(f"Invalid UCI move: {uci}")
        
        # If we have the same position but now with move history
        if fresh_board.fen().split(' ')[0] == board.fen().split(' ')[0]:
            # Use the board with move history
            board = fresh_board
        else:
            print("Board position doesn't match after replaying moves")
    
    return board


def save_board(board: chess.Board):
    """Save the board's FEN and move history back into session."""
    session['board_fen'] = board.fen()
    
    # Store UCI representation of all moves for debugging
    move_uci_list = [move.uci() for move in board.move_stack]
    session['move_history'] = move_uci_list


def get_engine():
    """Return the current engine (AI model) from session."""
    # Always recreate the engine from the model_name
    # This avoids session serialization errors
    model_name = session.get('model_name', '9M')
    engine = engine_constants.ENGINE_BUILDERS[model_name]()
    return engine


def get_highlighted_squares():
    return session.get('highlighted_squares', {})


def set_highlighted_squares(sq_dict):
    session['highlighted_squares'] = sq_dict


# ----------------------------------------------------------------------
# Helper: convert board to base64-encoded PNG
# ----------------------------------------------------------------------
def render_board_image(highlight_last_move=True):
    board = load_board()
    last_move = board.peek() if board.move_stack else None

    # Convert the session's highlight dictionary keys (string) back to int
    highlighted_squares = {int(k): v for k, v in get_highlighted_squares().items()}
    
    # Set orientation based on player color
    player_color = session.get('player_color', chess.WHITE)
    orientation = player_color

    # Generate SVG
    svg_data = chess.svg.board(
        board=board,
        fill=highlighted_squares,
        size=700,
        lastmove=last_move if highlight_last_move else None,
        orientation=orientation,
        style=(
            " .square.light { fill: #f0d9b5; }"
            " .square.dark { fill: #b58863; }"
        )
    )

    # Convert to PNG
    png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

    # Encode as base64
    encoded = base64.b64encode(png_data).decode('utf-8')
    return encoded


# ----------------------------------------------------------------------
# AI / attention visualization
# ----------------------------------------------------------------------
def ai_make_move():
    board = load_board()
    if not board.is_game_over():
        engine = get_engine()
        # AI chooses a move
        ai_move = engine.play(board)
        # Visualize attention
        visualize_attention(ai_move)
        # Make the move
        board.push(ai_move)
        save_board(board)


def visualize_attention(move: chess.Move):
    board = load_board()
    engine = get_engine()

    # 1) Tokenize the board FEN plus the chosen move and the dummy return bucket.
    tokenized_fen = tokenizer.tokenize(board.fen()).astype(np.int32)
    action = np.array([utils.MOVE_TO_ACTION[move.uci()]], dtype=np.int32)
    dummy_r = np.array([0], dtype=np.int32)  # Dummy return token

    sequence = np.concatenate([tokenized_fen, action, dummy_r])[None, :]  # shape [1,79]

    # 2) Inference
    logits, attention_weights, hidden_states = engine.predict_fn(sequence)

    # 3) Extract last-token-to-board-squares attention
    last_token_attention = [
        layer[0, :, 78, 1:65]  # shape = [num_heads, 64]
        for layer in attention_weights
    ]

    # 4) Average over heads/layers, then normalize
    avg_attention_per_layer = [np.mean(heads, axis=0) for heads in last_token_attention]
    avg_attention = np.mean(avg_attention_per_layer, axis=0)  # shape [64]

    attn_min, attn_max = avg_attention.min(), avg_attention.max()
    if attn_max > attn_min:
        attn_normalized = (avg_attention - attn_min) / (attn_max - attn_min)
    else:
        attn_normalized = np.zeros_like(avg_attention)

    # 5) Convert from FEN-index to python-chess indexing
    def fen_token_index_to_square(i):
        row = i // 8
        col = i % 8
        return 8 * (7 - row) + col

    # Build highlight dictionary
    high = {}
    for i, weight in enumerate(attn_normalized):
        if weight > 0.1:  # highlight squares w/ "significant" attention
            sq = fen_token_index_to_square(i)
            alpha = weight * 0.7
            high[sq] = f"rgba(255, 0, 0, {alpha})"

    # Also highlight the move
    high[move.from_square] = "rgba(0, 200, 0, 0.5)"
    high[move.to_square] = "rgba(0, 200, 0, 0.5)"
    set_highlighted_squares(high)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    """Main page: show board, controls, and move history."""
    if 'board_fen' not in session:
        init_game_state()

    board = load_board()
    move_stack = list(board.move_stack)
    player_color = session.get('player_color', chess.WHITE)

    # Check if AI should move
    ai_should_move = board.turn != player_color and not board.is_game_over()
    
    # Prepare move history lines
    moves_formatted = []
    for i in range(0, len(move_stack), 2):
        move_num = i // 2 + 1
        white_move = move_stack[i].uci() if i < len(move_stack) else ""
        black_move = ""
        if i + 1 < len(move_stack):
            black_move = move_stack[i+1].uci()
        moves_formatted.append(f"{move_num}. {white_move} {black_move}")
    
    # Prepare status
    if board.is_checkmate():
        status = f"Checkmate! {'Black' if board.turn else 'White'} wins!"
    elif board.is_stalemate():
        status = "Stalemate!"
    elif board.is_check():
        turn_color = "White" if board.turn else "Black"
        status = f"{turn_color} is in check!"
    else:
        turn_color = "White" if board.turn else "Black"
        status = f"{turn_color}'s turn"

    # Render the board as base64
    board_png = render_board_image()

    # Current model
    model_name = session.get('model_name', '9M')
    
    # For JavaScript
    ai_to_move = ai_should_move

    # We'll do a simple HTML render with an inline template
    # For bigger projects, you'd move this to a .html file in templates/
    html_page = render_template_string(
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Searchless Chess</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                :root {
                    --primary: #2c3e50;
                    --secondary: #3498db;
                    --accent: #e74c3c;
                    --background: #f7f9fc;
                    --card-bg: #ffffff;
                    --text: #333333;
                    --text-light: #7f8c8d;
                    --border: #e0e6ed;
                    --success: #2ecc71;
                    --shadow: 0 4px 12px rgba(0,0,0,0.08);
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Poppins', sans-serif;
                    background-color: var(--background);
                    color: var(--text);
                    margin: 0;
                    padding: 24px;
                    min-height: 100vh;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                header {
                    margin-bottom: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                
                h1 {
                    font-weight: 600;
                    font-size: 28px;
                    color: var(--primary);
                }
                
                .status {
                    display: inline-block;
                    padding: 8px 16px;
                    border-radius: 50px;
                    background-color: var(--primary);
                    color: white;
                    font-weight: 500;
                    font-size: 14px;
                    box-shadow: var(--shadow);
                }
                
                .status.check {
                    background-color: var(--accent);
                }
                
                .status.game-over {
                    background-color: var(--success);
                }
                
                .main-content {
                    display: flex;
                    gap: 30px;
                    flex-wrap: wrap;
                }
                
                #board-container {
                    position: relative;
                    width: 700px;
                    height: 700px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: var(--shadow);
                }
                
                #chessboard {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 700px;
                    height: 700px;
                }
                
                .square-overlay {
                    position: absolute;
                    width: calc(700px / 8);
                    height: calc(700px / 8);
                    cursor: pointer;
                }
                
                #sidebar {
                    flex: 1;
                    min-width: 300px;
                }
                
                .card {
                    background-color: var(--card-bg);
                    border-radius: 12px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: var(--shadow);
                }
                
                .card-header {
                    font-weight: 600;
                    font-size: 16px;
                    margin-bottom: 15px;
                    color: var(--primary);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                
                .model-group {
                    display: flex;
                    gap: 15px;
                    margin-bottom: 10px;
                }
                
                .model-option {
                    flex: 1;
                    position: relative;
                }
                
                .model-option input {
                    position: absolute;
                    opacity: 0;
                    width: 0;
                    height: 0;
                }
                
                .model-option label {
                    display: block;
                    background-color: var(--border);
                    padding: 10px;
                    text-align: center;
                    border-radius: 8px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .model-option input:checked + label {
                    background-color: var(--secondary);
                    color: white;
                    box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
                }
                
                .model-option:hover label {
                    background-color: var(--secondary);
                    opacity: 0.8;
                    color: white;
                }
                
                .move-history {
                    height: 250px;
                    overflow-y: auto;
                    border-radius: 8px;
                    background-color: var(--background);
                    scrollbar-width: thin;
                    font-family: 'Courier New', monospace;
                }
                
                .move-history::-webkit-scrollbar {
                    width: 6px;
                    background-color: var(--background);
                }
                
                .move-history::-webkit-scrollbar-thumb {
                    background-color: var(--border);
                    border-radius: 6px;
                }
                
                .move-history table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }
                
                .move-history th {
                    position: sticky;
                    top: 0;
                    background-color: var(--primary);
                    color: white;
                    text-align: left;
                    padding: 12px;
                }
                
                .move-history td {
                    padding: 10px 12px;
                    border-bottom: 1px solid var(--border);
                }
                
                .move-history tr:hover td {
                    background-color: rgba(52, 152, 219, 0.1);
                }
                
                .button-group {
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }
                
                .btn {
                    flex: 1;
                    min-width: 80px;
                    padding: 12px 16px;
                    background-color: var(--secondary);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-family: 'Poppins', sans-serif;
                    font-weight: 500;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
                }
                
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
                }
                
                .btn.new-game {
                    background-color: var(--success);
                    box-shadow: 0 2px 6px rgba(46, 204, 113, 0.3);
                }
                
                .btn.new-game:hover {
                    box-shadow: 0 4px 12px rgba(46, 204, 113, 0.4);
                }
                
                .btn.undo {
                    background-color: var(--primary);
                    box-shadow: 0 2px 6px rgba(44, 62, 80, 0.3);
                }
                
                .btn.undo:hover {
                    box-shadow: 0 4px 12px rgba(44, 62, 80, 0.4);
                }
                
                .btn.hint {
                    background-color: var(--accent);
                    box-shadow: 0 2px 6px rgba(231, 76, 60, 0.3);
                }
                
                .btn.hint:hover {
                    box-shadow: 0 4px 12px rgba(231, 76, 60, 0.4);
                }
                
                .btn.flip {
                    background-color: #8e44ad;
                    box-shadow: 0 2px 6px rgba(142, 68, 173, 0.3);
                }
                
                .btn.flip:hover {
                    box-shadow: 0 4px 12px rgba(142, 68, 173, 0.4);
                }
                
                .attention-info {
                    color: var(--text-light);
                    font-size: 14px;
                    line-height: 1.5;
                }
                
                .marker {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 6px;
                    vertical-align: middle;
                }
                
                .marker.red {
                    background-color: rgba(231, 76, 60, 0.7);
                }
                
                .marker.green {
                    background-color: rgba(0, 200, 0, 0.5);
                }
                
                @media (max-width: 1100px) {
                    .main-content {
                        justify-content: center;
                    }
                    
                    #board-container {
                        width: 600px;
                        height: 600px;
                    }
                    
                    #chessboard {
                        width: 600px;
                        height: 600px;
                    }
                    
                    .square-overlay {
                        width: calc(600px / 8);
                        height: calc(600px / 8);
                    }
                }
                
                @media (max-width: 640px) {
                    body {
                        padding: 16px;
                    }
                    
                    header {
                        flex-direction: column;
                        align-items: flex-start;
                        gap: 10px;
                    }
                    
                    .main-content {
                        gap: 20px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>Searchless Chess</h1>
                    <div class="status {% if 'check' in status.lower() %}check{% elif 'mate' in status.lower() or 'stale' in status.lower() %}game-over{% endif %}">{{ status }}</div>
                </header>
                
                <!-- Hidden fields for JavaScript -->
                <input type="hidden" id="ai_to_move" value="{{ 'true' if ai_to_move else 'false' }}">
                <input type="hidden" id="player_color" value="{{ 'white' if player_color == 1 else 'black' }}">
                
                <div class="main-content">
                    <!-- Board container -->
                    <div id="board-container">
                        <!-- Chessboard image -->
                        <img id="chessboard" src="data:image/png;base64,{{ board_png }}" alt="Chess board" />
                        <!-- 8x8 overlay squares to detect clicks -->
                        {% for rank in range(8) %}
                            {% for file in range(8) %}
                                {% set sq_x = file * (700/8) %}
                                {% set sq_y = rank * (700/8) %}
                                <div 
                                    class="square-overlay"
                                    style="left: {{sq_x}}px; top: {{sq_y}}px;"
                                    onclick="onSquareClick({{ 7-file if player_color == 0 else file }}, {{ rank if player_color == 0 else 7-rank }})"></div>
                            {% endfor %}
                        {% endfor %}
                    </div>

                    <div id="sidebar">
                        <!-- Model selection -->
                        <div class="card">
                            <div class="card-header">Model Selection</div>
                            <form action="/change_model" method="post" id="modelForm">
                                <div class="model-group">
                                    <div class="model-option">
                                        <input type="radio" id="model9M" name="model" value="9M" 
                                        {% if model_name == '9M' %}checked{% endif %} 
                                        onchange="this.form.submit()">
                                        <label for="model9M">9M</label>
                                    </div>
                                    <div class="model-option">
                                        <input type="radio" id="model136M" name="model" value="136M" 
                                        {% if model_name == '136M' %}checked{% endif %} 
                                        onchange="this.form.submit()">
                                        <label for="model136M">136M</label>
                                    </div>
                                    <div class="model-option">
                                        <input type="radio" id="model270M" name="model" value="270M" 
                                        {% if model_name == '270M' %}checked{% endif %} 
                                        onchange="this.form.submit()">
                                        <label for="model270M">270M</label>
                                    </div>
                                </div>
                            </form>
                        </div>

                        <!-- Move history -->
                        <div class="card">
                            <div class="card-header">Move History</div>
                            <div class="move-history">
                                <table>
                                    <tr><th>Moves</th></tr>
                                    {% if moves_formatted %}
                                        {% for line in moves_formatted %}
                                            <tr><td>{{ line }}</td></tr>
                                        {% endfor %}
                                    {% else %}
                                        <tr><td style="color: var(--text-light); font-style: italic;">No moves yet</td></tr>
                                    {% endif %}
                                </table>
                            </div>
                        </div>

                        <!-- Game controls -->
                        <div class="card">
                            <div class="card-header">Game Controls</div>
                            <div class="button-group">
                                <button class="btn new-game" onclick="newGame()">New Game</button>
                                <button class="btn undo" onclick="undoMove()">Undo</button>
                                <button class="btn hint" onclick="showHint()">Hint</button>
                                <button class="btn flip" onclick="flipBoard()">Flip Board</button>
                            </div>
                        </div>

                        <!-- Explanation for attention -->
                        <div class="card">
                            <div class="card-header">Attention Visualization</div>
                            <div class="attention-info">
                                <p><span class="marker red"></span> AI's attention (darker = more attention)</p>
                                <p><span class="marker green"></span> AI's selected move</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <script>
            // Check if AI needs to move on page load
            document.addEventListener('DOMContentLoaded', function() {
                const aiToMove = document.getElementById('ai_to_move').value === 'true';
                
                if (aiToMove) {
                    // Make AI move immediately after page loads showing player's move
                    fetch('/ai_move', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'ok') {
                            window.location.reload();
                        }
                    });
                }
                
                // Scroll to bottom of move history
                const moveHistory = document.querySelector('.move-history');
                moveHistory.scrollTop = moveHistory.scrollHeight;
            });

            function onSquareClick(file, rank) {
                fetch(`/select_square/${file}/${rank}`, {
                    method: 'POST'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'ok') {
                        window.location.reload();
                    } else {
                        // e.g. "illegal move"
                        window.location.reload();
                    }
                })
                .catch(err => console.log(err));
            }

            function newGame() {
                fetch("/new_game", { method: "POST" })
                .then(() => window.location.reload());
            }

            function undoMove() {
                fetch("/undo", { method: "POST" })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'ok') {
                        window.location.reload();
                    } else {
                        alert("Cannot undo further.");
                    }
                });
            }

            function showHint() {
                fetch("/hint", { method: "POST" })
                .then(response => response.json())
                .then(data => {
                    window.location.reload();
                });
            }
            
            function flipBoard() {
                fetch("/flip_board", { method: "POST" })
                .then(response => response.json())
                .then(data => {
                    window.location.reload();
                });
            }
            </script>
        </body>
        </html>
        """,
        status=status,
        board_png=board_png,
        moves_formatted=moves_formatted,
        model_name=model_name,
        ai_to_move=ai_to_move,
        player_color=player_color
    )
    return html_page


@app.route("/select_square/<int:file>/<int:rank>", methods=["POST"])
def select_square(file, rank):
    """Handle user clicks on squares. 
       In Python-chess indexing: square = chess.square(file, rank).
    """
    board = load_board()
    player_color = session.get('player_color', chess.WHITE)

    # If the game is over or it's not the player's turn, do nothing
    if board.is_game_over() or board.turn != player_color:
        return jsonify({"status": "ok"})

    square = chess.square(file, rank)
    from_sq = session.get('from_square', None)

    if from_sq is None:
        # First click: select piece
        piece = board.piece_at(square)
        if piece and piece.color == player_color:
            session['from_square'] = square
            # Highlight
            highlight_squares = {}
            highlight_squares[square] = "rgba(0, 255, 0, 0.5)"
            # highlight legal moves
            for mv in board.legal_moves:
                if mv.from_square == square:
                    highlight_squares[mv.to_square] = "rgba(0, 0, 255, 0.3)"
            set_highlighted_squares(highlight_squares)
        return jsonify({"status": "ok"})
    else:
        # Attempt move
        move = chess.Move(from_sq, square)

        # Check promotion
        if (board.piece_at(from_sq) and
            board.piece_at(from_sq).piece_type == chess.PAWN and
            ((square >= 56 and board.turn) or
             (square <= 7 and not board.turn))):
            move.promotion = chess.QUEEN

        if move in board.legal_moves:
            board.push(move)
            save_board(board)
            session['from_square'] = None
            set_highlighted_squares({})  # Clear highlight
            
            # Flag that AI needs to move after player sees their move
            return jsonify({"status": "ok"})
        else:
            # Invalid move
            session['from_square'] = None
            set_highlighted_squares({})
            return jsonify({"status": "invalid"})


@app.route("/ai_move", methods=["POST"])
def make_ai_move():
    """Make the AI move as a separate step after the player has seen their move."""
    board = load_board()
    player_color = session.get('player_color', chess.WHITE)
    
    # Check if it's the AI's turn
    if board.turn != player_color and not board.is_game_over():
        ai_make_move()
        return jsonify({"status": "ok"})
    return jsonify({"status": "not_needed"})


@app.route("/new_game", methods=["POST"])
def new_game():
    """Reset the game to starting position."""
    init_game_state()
    return ("", 204)


@app.route("/undo", methods=["POST"])
def undo():
    """Undo last two moves if possible."""
    board = load_board()
    if len(board.move_stack) >= 2:
        board.pop()
        board.pop()
        save_board(board)
        set_highlighted_squares({}) 
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "cannot"})


@app.route("/hint", methods=["POST"])
def hint():
    """Show an AI suggestion for user's next move."""
    board = load_board()
    player_color = session.get('player_color', chess.WHITE)
    
    if (not board.is_game_over()) and board.turn == player_color:
        engine = get_engine()
        hint_move = engine.play(board)
        # Highlight the suggested move
        highlight_squares = {
            hint_move.from_square: "rgba(255, 215, 0, 0.5)",
            hint_move.to_square: "rgba(255, 215, 0, 0.5)"
        }
        set_highlighted_squares(highlight_squares)
        return jsonify({"hint": hint_move.uci()})
    else:
        return jsonify({"hint": "No hint available"})


@app.route("/flip_board", methods=["POST"])
def flip_board():
    """Switch roles between player and AI."""
    # Clear any active selection
    session['from_square'] = None
    set_highlighted_squares({})
    
    # Flip player color (WHITE=1, BLACK=0)
    current_color = session.get('player_color', chess.WHITE)
    new_color = chess.BLACK if current_color == chess.WHITE else chess.WHITE
    session['player_color'] = new_color
    
    return jsonify({"status": "ok"})


@app.route("/change_model", methods=["POST"])
def change_model():
    """Update which AI model is used."""
    new_model = request.form.get("model", "9M")
    try:
        session['model_name'] = new_model
        flash(f"Changed model to {new_model}")
    except Exception as e:
        flash(f"Failed to load {new_model} model: {e}")
        session['model_name'] = "9M"
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

'''
Before running:
cd searchless_chess/src
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
export PYTHONPATH=$(pwd)/../..
'''
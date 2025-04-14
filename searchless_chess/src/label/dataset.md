# Chess Position Dataset Documentation

## Overview
This document describes the generation process for the chess position dataset used in the FischerGPT project. The dataset consists of chess positions in FEN notation with binary labels for various chess concepts.

## Source Data
- **Source**: Lichess PGN games
- **Sample Size**: Up to 50,000 games processed
- **Format**: Positions stored as FEN strings with binary labels (0/1)

## Generated Datasets
Six distinct datasets are created:

1. **Material Difference** (`material_difference.txt`)
   - **Label 1**: Material difference > 3 pawns
   - **Label 0**: Material difference ≤ 3 pawns

2. **Check Status** (`in_check.txt`)
   - **Label 1**: King is in check
   - **Label 0**: King is not in check

3. **Passed Pawns** (`passed_pawn.txt`)
   - **Label 1**: Position has at least one passed pawn
   - **Label 0**: No passed pawns exist

4. **Bishop Pair** (`bishop_pair.txt`)
   - **Label 1**: Either side has both bishops
   - **Label 0**: Neither side has both bishops

5. **Mobility** (`high_mobility.txt`)
   - **Label 1**: Side to move has > 25 legal moves
   - **Label 0**: Side to move has ≤ 25 legal moves

6. **Pinned Pieces** (`pinned_piece.txt`)
   - **Label 1**: Position has at least one pinned piece
   - **Label 0**: No pinned pieces exist

## Dataset Generation Process

### Position Sampling Strategy
To maximize diversity and avoid redundant positions:

1. **Game Phase Sampling**:
   - **Early Game**: Moves 10-20
   - **Middle Game**: Moves 20-40
   - **End Game**: Moves 40+
   - Each phase contributes approximately equally to samples

2. **Position Selection**:
   - 5 positions sampled per game
   - First 10 moves (opening theory) skipped
   - Random sampling within each phase
   - No consecutive positions to avoid similarity

3. **Uniqueness**:
   - Duplicate FEN positions are filtered out
   - Each position is recorded only once

### Class Balancing
To ensure balanced datasets (approximately 50/50 split):

1. **Label Tracking**:
   - Counters maintain running totals for each label (0/1)
   - For each concept (material, check, etc.)

2. **Selection Algorithm**:
   - For each position, label is computed
   - Position is written to file only if:
     - Current label is underrepresented (or equal) compared to other label
   - This ensures approximately equal distribution of labels

### Implementation Details

#### Passed Pawn Detection
Pawn considered passed when:
- No enemy pawns are on the same file ahead of it
- No enemy pawns are on adjacent files ahead of it
- Direction is determined by pawn color (white: up, black: down)

#### Bishop Pair Detection
Position has bishop pair when:
- Either white or black has at least 2 bishops

#### Mobility Calculation
- Count legal moves available to the side to move
- High mobility threshold: > 25 legal moves

#### Pinned Piece Detection
- Piece is pinned if its movement would expose the king to attack
- Uses chess.Board.is_pinned() method

## File Format
Each dataset file contains lines formatted as:
```
<FEN>\t<LABEL>
```
Where:
- FEN is the Forsyth-Edwards Notation string
- LABEL is 0 or 1
- Values are tab-separated 

Total unique positions: 227057
Material difference positions: 68271
In check positions: 30913
Passed pawn positions: 88046
Bishop pair positions: 114009
High mobility positions: 79623
Pinned piece positions: 70529
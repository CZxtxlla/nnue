import math
import struct
from datasets import load_dataset

# Constants
MAX_ACTIVE = 32
SIGMOID_K = 410.0
POSITIONS_TO_DOWNLOAD = 100_000_000  # Set your target here

# Mappings
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4,
    'p': 5, 'n': 6, 'b': 7, 'r': 8, 'q': 9
}

def flip_sq(sq):
    return sq ^ 56

def flip_piece(p_type):
    return (p_type + 5) % 10

def cp_to_prob(centipawns):
    return 1.0 / (1.0 + math.exp(-centipawns / SIGMOID_K))

def process_position(fen_line, score_line):
    # 1. Parse FEN first so we have the 'stm' variable
    parts = fen_line.split()
    if len(parts) < 2: return None
    
    board = parts[0]
    stm = 1 if parts[1] == 'b' else 0
    
    # 2. Parse Score
    if '#' in score_line:
        target_prob = 0.0 if '#-' in score_line else 1.0
    else:
        try:
            pawns = float(score_line)
            target_prob = cp_to_prob(pawns * 100.0)
        except ValueError:
            return None # Skip malformed scores
        
    # 3. Convert Absolute probability to Active Player probability
    if stm == 1:
        target_prob = 1.0 - target_prob

    # 4. Extract pieces and squares
    wk_sq = -1
    bk_sq = -1
    pieces = []
    squares = []
    
    sq = 56
    for char in board:
        if char == '/':
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        elif char == 'K':
            wk_sq = sq
            sq += 1
        elif char == 'k':
            bk_sq = sq
            sq += 1
        elif char in PIECE_MAP:
            pieces.append(PIECE_MAP[char])
            squares.append(sq)
            sq += 1

    if wk_sq == -1 or bk_sq == -1:
        return None

    # 5. Compute HalfKP Indices
    white_indices = [-1] * MAX_ACTIVE
    black_indices = [-1] * MAX_ACTIVE

    for i in range(len(pieces)):
        if i >= MAX_ACTIVE: break
        
        p = pieces[i]
        s = squares[i]

        # White Perspective
        white_indices[i] = (wk_sq * 641) + (p * 64) + s

        # Black Perspective
        b_p = flip_piece(p)
        b_s = flip_sq(s)
        flipped_bk_sq = flip_sq(bk_sq)
        black_indices[i] = (flipped_bk_sq * 641) + (b_p * 64) + b_s

    # 6. Pack into C-compatible binary struct
    struct_format = f"<{MAX_ACTIVE}i {MAX_ACTIVE}i i f"
    
    binary_data = struct.pack(
        struct_format,
        *white_indices,
        *black_indices,
        stm,
        target_prob
    )
    
    return binary_data

def process_position(fen_line, score_line):
    # 1. Parse FEN first so we have the 'stm' variable
    parts = fen_line.split()
    if len(parts) < 2: return None
    
    board = parts[0]
    stm = 1 if parts[1] == 'b' else 0
    
    # 2. Parse Score
    if '#' in score_line:
        target_prob = 0.0 if '#-' in score_line else 1.0
    else:
        try:
            pawns = float(score_line)
            target_prob = cp_to_prob(pawns * 100.0)
        except ValueError:
            return None # Skip malformed scores
        
    # 3. Convert Absolute probability to Active Player probability
    if stm == 1:
        target_prob = 1.0 - target_prob

    # 4. Extract pieces and squares
    wk_sq = -1
    bk_sq = -1
    pieces = []
    squares = []
    
    sq = 56
    for char in board:
        if char == '/':
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        elif char == 'K':
            wk_sq = sq
            sq += 1
        elif char == 'k':
            bk_sq = sq
            sq += 1
        elif char in PIECE_MAP:
            pieces.append(PIECE_MAP[char])
            squares.append(sq)
            sq += 1

    if wk_sq == -1 or bk_sq == -1:
        return None

    # 5. Compute HalfKP Indices
    white_indices = [-1] * MAX_ACTIVE
    black_indices = [-1] * MAX_ACTIVE

    for i in range(len(pieces)):
        if i >= MAX_ACTIVE: break
        
        p = pieces[i]
        s = squares[i]

        # White Perspective
        white_indices[i] = (wk_sq * 641) + (p * 64) + s

        # Black Perspective
        b_p = flip_piece(p)
        b_s = flip_sq(s)
        flipped_bk_sq = flip_sq(bk_sq)
        black_indices[i] = (flipped_bk_sq * 641) + (b_p * 64) + b_s

    # 6. Pack into C-compatible binary struct
    struct_format = f"<{MAX_ACTIVE}i {MAX_ACTIVE}i i f"
    
    binary_data = struct.pack(
        struct_format,
        *white_indices,
        *black_indices,
        stm,
        target_prob
    )
    
    return binary_data

def main():
    print("Connecting to Hugging Face stream...")
    # streaming=True allows us to process data without downloading 100GB to disk
    dataset = load_dataset("lukesalamone/gigafish-3.8b-d10", split="train", streaming=True)
    
    output_file = "bootstrapping_data.bin"
    positions_saved = 0
    
    with open(output_file, "wb") as f:
        print(f"Streaming and compiling data into {output_file}...")
        
        for row in dataset:
            # The dataset is already perfectly separated into columns!
            fen_line = row['fen'].strip()
            score_line = row['eval'].strip()
            
            bin_data = process_position(fen_line, score_line)
            
            if bin_data:
                f.write(bin_data)
                positions_saved += 1
                
                if positions_saved % 50_000 == 0:
                    print(f"[{positions_saved}/{POSITIONS_TO_DOWNLOAD}] positions compiled...")
                    
            if positions_saved >= POSITIONS_TO_DOWNLOAD:
                break
                
    print(f"\nDone! Successfully packed {positions_saved} positions.")

if __name__ == "__main__":
    main()
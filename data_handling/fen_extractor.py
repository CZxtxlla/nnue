import chess.pgn
import random
import os

def generate_quality_fens(pgn_file_path, output_file_path, max_fens=1000000):
    fens_collected = 0

    print(f"Opening PGN file: {pgn_file_path}")

    with open(pgn_file_path, "r", encoding="utf-8") as pgn_file, \
         open(output_file_path, "w", encoding="utf-8") as out_file:
             
        while fens_collected < max_fens:
            # Read the next game from the massive PGN file
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                print("Reached the end of the PGN file.")
                break 
            
            board = game.board()
            
            # Play through the game move by move
            for ply_number, move in enumerate(game.mainline_moves()):
                board.push(move)
                
                # QUALITY FILTERS
                
                # Skip the first 16 plies (8 full moves) to avoid opening theory
                if ply_number < 16:
                    continue
                    
                # Skip positions where the king is in check
                if board.is_check():
                    continue
                    
                # Skip if the game is already over (mate or stalemate)
                if board.is_game_over():
                    break
                    
                # Anti-Correlation, Only take ~10% of the eligible positions from this game
                if random.random() > 0.10:
                    continue
                
                # If it passes all filters, write the FEN to text file
                out_file.write(board.fen() + "\n")
                fens_collected += 1
                
                if fens_collected >= max_fens:
                    break # Break out of the move loop
            
            if fens_collected % 50_000 == 0:
                print(f"Progress: Extracted {fens_collected} FENs...")
                
    print(f"Finished! Successfully saved {fens_collected} FENs to {output_file_path}")

if __name__ == '__main__':
    # path to lichess pgn file
    input_pgn = "lichess_db_standard_rated_2015-05.pgn"
    
    # This is the file you will read from in your data_generation.py script
    output_fens = "filtered_fens.txt"

    target_amount = 1000000 
    
    if not os.path.exists(input_pgn):
        print(f"Error: Could not find {input_pgn}. Please download one and update the path.")
    else:
        generate_quality_fens(input_pgn, output_fens, target_amount)
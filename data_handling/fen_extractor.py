import chess.pgn
import random
import os

def generate_quality_fens(pgn_file_paths, output_file_path, max_fens=1000000):
    fens_collected = 0

    # open output file
    with open(output_file_path, "w", encoding="utf-8") as out_file:
        
        # loop through each PGN file in the list provided
        for pgn_path in pgn_file_paths:
            
            # If we already hit our target, stop opening new files
            if fens_collected >= max_fens:
                break
                
            if not os.path.exists(pgn_path):
                print(f"Warning: Could not find {pgn_path}. Skipping to next file...")
                continue
                
            print(f"\n--- Opening PGN file: {pgn_path} ---")
            
            # open currentr PGN file
            with open(pgn_path, "r", encoding="utf-8") as pgn_file:
                while fens_collected < max_fens:
                    # Read the next game
                    game = chess.pgn.read_game(pgn_file)
                    
                    if game is None:
                        print(f"Reached the end of {pgn_path}.")
                        break # Break the while loop to move to next file
                    
                    board = game.board()
                    
                    # Play through the game move by move
                    for ply_number, move in enumerate(game.mainline_moves()):
                        board.push(move)
                        
                        # QUALITY FILTERS
                        if ply_number < 16:
                            continue
                        if board.is_check():
                            continue
                        if board.is_game_over():
                            break
                        if random.random() > 0.10:
                            continue
                        
                        # Write the fen
                        out_file.write(board.fen() + "\n")
                        fens_collected += 1
                        
                        if fens_collected % 50_000 == 0:
                            print(f"Progress: Extracted {fens_collected} FENs...")
                            
                        if fens_collected >= max_fens:
                            break # break out of the move loop
                            
    print(f"\nFinished! Successfully saved {fens_collected} FENs to {output_file_path}")

if __name__ == '__main__':
    # list of datasets
    input_pgns = [
        "data/lichess_elite_2021-11.pgn",
        "data/lichess_elite_2021-10.pgn",
        "data/lichess_elite_2021-09.pgn",
        "data/lichess_elite_2021-08.pgn",
        "data/lichess_elite_2021-07.pgn",
        "data/lichess_elite_2021-06.pgn",
        "data/lichess_elite_2021-05.pgn",
        "data/lichess_elite_2021-04.pgn",
        "data/lichess_elite_2021-03.pgn",
        "data/lichess_elite_2021-02.pgn",
        "data/lichess_elite_2021-01.pgn"
    ]
    
    output_fens = "data/filtered_fens.txt"

    target_amount = 50_000_000 
    
    generate_quality_fens(input_pgns, output_fens, target_amount)
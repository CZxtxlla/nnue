import chess
import chess.engine
import struct
import random
import concurrent.futures
import multiprocessing
import os
import shutil

# Struct format: 
# < (little endian)
# h (int16 eval)
# HHH (uint16 w, d, l)
# BB (uint8 stm, num_features)
# 32H (uint16 array of 32 features)
STRUCT_FORMAT = '<h 3H 2B 32H'

def concatenate_bin_files(file_list, final_output_name):
    print(f"\nConcatenating {len(file_list)} files into {final_output_name}...")
    
    with open(final_output_name, 'wb') as outfile:
        for bin_file in file_list:
            if os.path.exists(bin_file):
                with open(bin_file, 'rb') as infile:
                    # copyfileobj efficiently streams data without maxing out RAM
                    shutil.copyfileobj(infile, outfile)
                
                # Delete the temporary worker file to save disk space
                os.remove(bin_file)
            else:
                print(f"Warning: {bin_file} not found.")
                
    print(f"Successfully created {final_output_name} and cleaned up temporary files.")

def get_features(board):
    "convert board to list of indices of active features"
    features = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            #0-5 for white, 6-11 for black
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = (piece.piece_type - 1) + color_offset

            feature_idx = (piece_idx * 64) + sq
            features.append(feature_idx)
    return features

def process_fen_chunk(engine_path, chunk_fens, worker_id):
    # each cpu core runs this independently
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": 32, "UCI_ShowWDL": True,})
    
    total_in_chunk = len(chunk_fens)
    print(f"Worker {worker_id}: engine started. Processing {total_in_chunk} FENs...", flush=True)
    
    # Create a unique file for this worker to prevent write collisions
    output_file = f"training_data_part_{worker_id}.bin"

    with open(output_file, 'wb') as f:
        # NEW: Use enumerate to track how many FENs this worker has processed
        for i, fen in enumerate(chunk_fens):
            
            #progress indicator
            if i > 0 and i % 5000 == 0:
                percent_done = (i / total_in_chunk) * 100
                print(f"Worker {worker_id} Progress: {i} / {total_in_chunk} ({percent_done:.1f}%)", flush=True)

            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            
            # Eval
            score = info["score"].white()
            if score.is_mate():
                eval_cp = 10000 if score.mate() > 0 else -10000
            else:
                eval_cp = score.score()
                
            # WDL Probabilities
            wdl = score.wdl()
            if wdl is not None:
                win, draw, loss = wdl
            else:
                win, draw, loss = 0, 0, 0
                
            # STM and Features
            stm = 0 if board.turn == chess.WHITE else 1
            features = get_features(board)
            num_features = len(features)
            
            # pad with 65535
            padded_features = features + [65535] * (32 - num_features)
            
            # Pack and Write
            try:
                packed_data = struct.pack(
                    STRUCT_FORMAT,
                    eval_cp, win, draw, loss, stm, num_features, *padded_features
                )
                f.write(packed_data)
            except struct.error as e:
                print(f"Worker {worker_id} failed on FEN {fen}: {e}", flush=True)

    # close the engine when the chunk is done
    engine.quit()
    print(f"Worker {worker_id}: FINISHED!", flush=True)
    return output_file


def generate_data_multicore(engine_path, fen_list, num_cores):
    # assign portions to cores
    chunk_size = (len(fen_list) // num_cores) + 1
    chunks = [fen_list[i:i + chunk_size] for i in range(0, len(fen_list), chunk_size)]

    print(f"Starting {num_cores} workers. Total FENs: {len(fen_list)}...")

    generated_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i, chunk in enumerate(chunks):
            if chunk: # Only submit if chunk is not empty
                futures.append(executor.submit(process_fen_chunk, engine_path, chunk, i))
            
        # Wait for all cores to finish and collect their filenames
        for future in concurrent.futures.as_completed(futures):
            generated_files.append(future.result())

    final_dataset_name = "training_data_final.bin"
    concatenate_bin_files(generated_files, final_dataset_name)
    print(f"Finished! Data written to: {final_dataset_name}")


if __name__ == '__main__':
    multiprocessing.set_start_method("fork")
    my_engine_path = "/home/cszit/stockfish/stockfish-ubuntu-x86-64-avx2"
    fen_file_path = "data/filtered_fens.txt"
    
    print(f"Loading FENs from {fen_file_path}...")
    
    # read file and remove duplicates
    with open(fen_file_path, "r") as f:
        # .strip() removes newlines, set() removes duplicates
        unique_fens = set(line.strip() for line in f if line.strip())
        
    # Convert back to a list to be sliced into chunks
    fens_to_evaluate = list(unique_fens)
    
    print(f"Successfully loaded {len(fens_to_evaluate)} UNIQUE positions.")
    
    # use all available cpu cores minus 1
    available_cores = max(1, multiprocessing.cpu_count() - 1)
    
    generate_data_multicore(my_engine_path, fens_to_evaluate, available_cores) 
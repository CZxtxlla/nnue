use std::fs::File;
use std::io::{BufWriter, Write};
use sfbinpack::CompressedTrainingDataEntryReader;

fn main() -> std::io::Result<()> {
    let in_file = File::open("data/small_chunk.binpack").expect("Failed to open binpack");
    let mut reader = CompressedTrainingDataEntryReader::new(in_file).unwrap();
    
    let records_per_file = 2_000_000;
    // 11 files, 10 for training one for validation
    let target_files = 11; 
    let max_total_records = records_per_file * target_files;

    let mut total_records = 0;
    let mut file_index = 0;
    
    // Initialize the first file
    let mut out_file = File::create(format!("training_data_part_{}.bin", file_index))?;
    let mut writer = BufWriter::new(out_file);

    println!("Starting translation...");

    while reader.has_next() && total_records < max_total_records {
        let entry = reader.next();

        // evaluation
        let eval: i16 = entry.score;
        writer.write_all(&eval.to_le_bytes())?;

        // wdl
        let mut win: u16 = 0;
        let mut draw: u16 = 0;
        let mut loss: u16 = 0;
        match entry.result {
            2 => win = 1000,
            1 => draw = 1000,
            _ => loss = 1000,
        }
        writer.write_all(&win.to_le_bytes())?;
        writer.write_all(&draw.to_le_bytes())?;
        writer.write_all(&loss.to_le_bytes())?;

        // side to move
        let stm: u8 = (entry.ply % 2) as u8; 
        writer.write_all(&[stm])?;

        // features
        let fen = entry.pos.fen().unwrap();
        let active_features = parse_fen_to_features(&fen);
        
        let num_features: u8 = active_features.len() as u8;
        writer.write_all(&[num_features])?;

        for i in 0..32 {
            if i < active_features.len() {
                writer.write_all(&active_features[i].to_le_bytes())?;
            } else {
                writer.write_all(&65535u16.to_le_bytes())?; 
            }
        }
        
        total_records += 1;

        // check if time to move to next file
        if total_records % records_per_file == 0 {
            writer.flush()?; // Save the current file
            println!("Finished writing training_data_part_{}.bin", file_index);
            
            file_index += 1;
            
            // open the next file
            if file_index < target_files {
                let new_out = File::create(format!("training_data_part_{}.bin", file_index))?;
                writer = BufWriter::new(new_out);
            }
        }
    }

    println!("Successfully translated {} records across {} files!", total_records, file_index);
    Ok(())
}

// helper to turn fen into feature indices
fn parse_fen_to_features(fen: &str) -> Vec<u16> {
    let mut features = Vec::new();
    let fen_board = fen.split_whitespace().next().unwrap();
    
    let mut rank = 7;
    let mut file = 0;
    
    for c in fen_board.chars() {
        if c == '/' {
            rank -= 1;
            file = 0;
        } else if c.is_digit(10) {
            file += c.to_digit(10).unwrap();
        } else {
            let square = rank * 8 + file;
            let piece_idx = match c {
                'P' => 0, 'N' => 1, 'B' => 2, 'R' => 3, 'Q' => 4, 'K' => 5,
                'p' => 6, 'n' => 7, 'b' => 8, 'r' => 9, 'q' => 10, 'k' => 11,
                _ => continue,
            };
            
            features.push((piece_idx * 64 + square) as u16);
            file += 1;
        }
    }
    features
}
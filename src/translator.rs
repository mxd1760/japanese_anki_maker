use std::path::PathBuf;

use ort::{session::Session, value::Tensor};
use sentencepiece_rs::SentencePieceProcessor;

const SOURCE_SPM_FILE: &str = "./fugumt_onnx/source.spm";
const TARGET_SPM_FILE: &str = "./fugumt_onnx/target.spm";
const ENCODER_MODEL_FILE: &str = "./fugumt_onnx/encoder_model.onnx";
const DECODER_MODEL_FILE: &str = "./fugumt_onnx/decoder_model.onnx";

pub struct Translator {
    spp: SentencePieceProcessor,
    tpp: SentencePieceProcessor,
    encoder_session: Session,
    decoder_session: Session,
}

impl Translator {
    pub fn new(models_dir:&PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let mut spp_file_path = models_dir.clone();
        spp_file_path.push(SOURCE_SPM_FILE);
        let spp = SentencePieceProcessor::open(spp_file_path)
            .map_err(|e| format!("Failed to load SentencePiece source model: {e}"))?;
        let mut tpp_file_path = models_dir.clone();
        tpp_file_path.push(TARGET_SPM_FILE);
        let tpp = SentencePieceProcessor::open(tpp_file_path)
            .map_err(|e| format!("failed to load SentencePiece target model: {e}"))?;
        let mut enc_file_path = models_dir.clone();
        enc_file_path.push(ENCODER_MODEL_FILE);
        let encoder_session = Session::builder()?.commit_from_file(enc_file_path)?;
        let mut dec_file_path = models_dir.clone();
        dec_file_path.push(DECODER_MODEL_FILE);
        let decoder_session = Session::builder()?.commit_from_file(dec_file_path)?;
        Ok(Self {
            spp,
            tpp,
            encoder_session,
            decoder_session,
        })
    }

    pub fn translate(&mut self, japanese_text: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // let pieces = self.spp
        //    .encode(&japanese_text)
        //    .map_err(|e| format!("Failed to tokenize text: {e}"))?;
        let raw_ids = self
            .spp
            .encode_to_ids(&japanese_text)
            .map_err(|e| format!("failed to generate ids: {e}"))?;
        //let decoded = self.spp.decode_ids(&raw_ids)?;
        let mut ids: Vec<i64> = raw_ids.into_iter().map(|id| id as i64).collect();
        ids.push(0); // Marian EOS token
        //println!(" pieces: {pieces:?}\n ids: {ids:?}\n decoded: {decoded:?}");

        let sequence_length = ids.len();

        let input_ids = Tensor::from_array(([1usize, sequence_length], ids))?;

        let attention_mask =
            Tensor::from_array(([1usize, sequence_length], vec![1i64; sequence_length]))?;

        let encoder_outputs = self.encoder_session.run(ort::inputs! {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
        })?;

        //let decoder_start_id = 32000i64;
        //let decoder_input = Tensor::from_array(([1usize, 1usize], vec![decoder_start_id]))?;

        let encoder_attention_mask =
            Tensor::from_array(([1usize, sequence_length], vec![1i64; sequence_length]))?;
        let encoder_hidden_states = encoder_outputs
            .get("last_hidden_state")
            .ok_or("Encoder did not produce last_hidden_state")?;

        let mut generated_ids: Vec<i64> = vec![32000]; // decoder_start_token_id
        let max_length = 100;

        for _step in 0..max_length {
            let decoder_length = generated_ids.len();

            let decoder_input =
                Tensor::from_array(([1usize, decoder_length], generated_ids.clone()))?;

            let decoder_outputs = self.decoder_session.run(ort::inputs! {
                "encoder_attention_mask" => encoder_attention_mask.clone(),
                "input_ids" => decoder_input,
                "encoder_hidden_states" => encoder_hidden_states,
            })?;

            let logits = decoder_outputs
                .get("logits")
                .ok_or("Decoder did not produce logits")?;

            let (_shape, data) = logits.try_extract_tensor::<f32>()?;

            // logits has shape [1, decoder_length, vocab_size].
            // We want the predictions for the LAST generated position.
            let vocab_size = 32001;
            let last_position = decoder_length - 1;
            let start = last_position * vocab_size;
            let end = start + vocab_size;

            let last_logits = &data[start..end];

            // Find the highest-scoring token.
            let next_id = last_logits
                .iter()
                .enumerate()
                // Don't generate the decoder start/padding token.
                .filter(|(id, _)| *id != 32000)
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(id, _)| id as i64)
                .ok_or("Could not find next token")?;

            //println!("step {:2}: token {}", step + 1, next_id);

            // EOS = 0
            if next_id == 0 {
                //println!("Reached EOS.");
                break;
            }

            generated_ids.push(next_id);
        }
        //println!("Generated IDs: {generated_ids:?}");

        let output_ids = &generated_ids[1..];

        let translation = self
            .tpp
            .decode_ids(&output_ids.iter().map(|id| *id as _).collect::<Vec<_>>())?;

        Ok(translation)
    }
}
use std::fs::File;
use std::io::prelude::*;

use furigana::Furigana;
use manga_ocr_rs::MangaOcr;
use ort::{session::Session, value::Tensor};
use sentencepiece_rs::SentencePieceProcessor;

struct Translator {
    spp: SentencePieceProcessor,
    tpp: SentencePieceProcessor,
    encoder_session: Session,
    decoder_session: Session,
}

#[derive(Clone)]
pub struct Note {
    japanese: String,
    furigana: Option<String>,
    english: String,
    image: Option<String>,
    audio: Option<String>,
    // tags: Vec<String>,
}

const SOURCE_SPM_FILE: &str = "./models/fugumt_onnx/source.spm";
const TARGET_SPM_FILE: &str = "./models/fugumt_onnx/target.spm";
const ENCODER_MODEL_FILE: &str = "./models/fugumt_onnx/encoder_model.onnx";
const DECODER_MODEL_FILE: &str = "./models/fugumt_onnx/decoder_model.onnx";

impl Translator {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let spp = SentencePieceProcessor::open(SOURCE_SPM_FILE)
            .map_err(|e| format!("Failed to load SentencePiece source model: {e}"))?;
        let tpp = SentencePieceProcessor::open(TARGET_SPM_FILE)
            .map_err(|e| format!("failed to load SentencePiece target model: {e}"))?;
        let encoder_session = Session::builder()?.commit_from_file(ENCODER_MODEL_FILE)?;
        let decoder_session = Session::builder()?.commit_from_file(DECODER_MODEL_FILE)?;
        Ok(Self {
            spp,
            tpp,
            encoder_session,
            decoder_session,
        })
    }

    pub fn translate(&mut self, japanese_text: &str) -> Result<String, Box<dyn std::error::Error>> {
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

impl Note {
    pub fn save_to_anki_text_file(notes: &[Self]) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create("output.txt")?;
        // write Anki headers
        file.write(
            r"#separator:Semicolon
#html:true
#columns:Expression;Reading;Meaning;Image_URI;Audio
#notetype:iKnow! Vocabulary
#deck:Japanese Anki Maker Tests

"
                .as_bytes(),
        )?;
        // write fields
        for i in notes {
            file.write(i.to_anki_text_line().as_bytes())?;
        }
        Ok(())
    }

    fn to_anki_text_line(&self) -> String {
        let jap: String = self.japanese.clone();
        let fur: String = match self.furigana.clone() {
            Some(v) => v,
            None => "".to_owned(),
        };
        let eng: String = self.english.clone();
        let img: String = match self.image.clone() {
            Some(v) => format!("<img src={:?}>", v),
            None => "".to_owned(),
        };
        let aud: String = match self.audio.clone() {
            Some(v) => format!("[sound:{:?}]", v),
            None => "".to_owned(),
        };

        format!("{};{};{};{};{}\n", jap, fur, eng, img, aud)
    }
}

const FILE_DIR: &str = "./test_images/";
const FILE_NAME: &str = "Screenshot 2025-04-20 125318.png";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = check_manga_ocr()?;
    let f = Furigana::minimal()?;
    let mut translator = Translator::new()?;

    // let test_text = "お前わもう死んでいる";
    let img = image::open(format!("{}{}", FILE_DIR, FILE_NAME))?;

    let text = ocr.recognize(&img)?;
    let furigana = f.to_hiragana(&text);
    // let text = "私は猫が好きです。";
    println!(" source: {},\n reading: {}", &text, furigana);

    let translation = translator.translate(&text)?;

    let mut notes = vec![];
    notes.push(Note {
        japanese: text,
        furigana: Some(furigana),
        english: translation.clone(),
        image: Some(FILE_NAME.to_owned()),
        audio: None,
    });
    Note::save_to_anki_text_file(&notes)?;

    println!(" translation: {translation}");

    Ok(())
}

fn check_manga_ocr() -> Result<MangaOcr, Box<dyn std::error::Error>> {
    let model_dir = manga_ocr_rs::default_model_dir();

    // First check that the expected files exist.
    let required_files = ["encoder_model.onnx", "decoder_model.onnx", "vocab.txt"];

    for file in required_files {
        let path = model_dir.join(file);

        if !path.exists() {
            return Err(format!(
                "Manga OCR model file is missing:\n{}\n\n\
                 Try running `cargo build` again or reinstalling the \
                 manga-ocr-rs models.",
                path.display()
            )
            .into());
        }
    }

    // Now actually try loading the models.
    let ocr = MangaOcr::new(&model_dir).map_err(|e| {
        format!(
            "Manga OCR model files were found, but could not be loaded \
                 correctly from:\n{}\n\n\
                 The model download may be incomplete or corrupted.\n\
                 Try deleting the model files and running `cargo build` again.\n\n\
                 Original error: {}",
            model_dir.display(),
            e
        )
    })?;

    Ok(ocr)
}

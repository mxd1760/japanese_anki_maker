use clap::Parser;
use std::{path::PathBuf, sync::Mutex};

use furigana::Furigana;
use manga_ocr_rs::MangaOcr;

use crate::{note::Note, translator::Translator};

mod progress_bar;
mod translator;
mod note;

#[derive(Parser)]
struct Args {
    #[arg(required = true, num_args = 1..)]
    files: Vec<PathBuf>,
    #[arg(short,long,default_value_t=4)]
    threads:usize
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = check_manga_ocr()?;
    let fg = Furigana::minimal()?;

    let args = Args::parse();

    let imgs = collect_images(&args.files)?;
    let name:Option<String> = args.files.get(0).unwrap().file_name().map(|s| s.to_owned().into_string().unwrap());

    let thread_pool_size = args.threads.min(imgs.len());
    let mut translator_vec = vec![];
    for _ in 0..thread_pool_size{
        translator_vec.push(Mutex::new(Translator::new()?))
    }

    println!();
    print!("Found {} files to process",imgs.len());
    let notes = Note::from_img_vec(&ocr, &fg, &translator_vec, &imgs)?;
    Note::save_to_anki_text_file(&notes, name)?;

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

fn collect_images(files: &[PathBuf]) -> Result<Vec<PathBuf>,Box<dyn std::error::Error>>{
    let mut imgs:Vec<PathBuf> = vec![];
    for item in files{
        if item.is_file(){
            imgs.push(item.clone())
        }else if item.is_dir(){
            for entry in std::fs::read_dir(item)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    imgs.push(path);
                }
            }
        }
    };
    Ok(imgs)
}


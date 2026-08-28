use manga_ocr_rs::MangaOcr;
use furigana::Furigana;
// use std::path::Path;



fn main() -> Result<(), Box<dyn std::error::Error>>{
    let ocr = check_manga_ocr()?;
    let f = Furigana::minimal()?;

    let img = image::open("./test_images/Screenshot 2025-04-20 123359.png")?;
    
    let text = ocr.recognize(&img)?;
    println!{" source: {},\n reading: {}",&text,f.to_hiragana(&text)}

    Ok(())
}



fn check_manga_ocr() -> Result<MangaOcr, Box<dyn std::error::Error>> {
    let model_dir = manga_ocr_rs::default_model_dir();

    // First check that the expected files exist.
    let required_files = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "vocab.txt",
    ];

    for file in required_files {
        let path = model_dir.join(file);

        if !path.exists() {
            return Err(format!(
                "Manga OCR model file is missing:\n{}\n\n\
                 Try running `cargo build` again or reinstalling the \
                 manga-ocr-rs models.",
                path.display()
            ).into());
        }
    }

    // Now actually try loading the models.
    let ocr = MangaOcr::new(&model_dir)
        .map_err(|e| {
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
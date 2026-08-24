use manga_ocr_rs::MangaOcr;


fn main() -> Result<(), Box<dyn std::error::Error>>{
    let ocr = MangaOcr::new(manga_ocr_rs::default_model_dir())?;
    
    let img = image::open("./test_images/Screenshot 2025-04-20 123359.png")?;
    
    let test = ocr.recognize(&img)?;
    println!("YES!!! \n {test}");

    Ok(())
}

use std::io::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use furigana::Furigana;
use manga_ocr_rs::MangaOcr;

use crate::progress_bar::ProgressBar;
use crate::translator::Translator;



#[derive(Clone)]
pub struct Note {
    japanese: String,
    furigana: Option<String>,
    english: String,
    image: Option<String>,
    audio: Option<String>,
    // tags: Vec<String>,
}

impl Note {
    pub fn save_to_anki_text_file(
        notes: &[Self],
        deck_name: Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::create("output.txt")?;
        // write Anki headers
        let deck = match deck_name {
            Some(v) => v,
            None => "Japanese Anki Maker Template".into(),
        };
        file.write(
            format!(
                r"#separator:Semicolon
#html:true
#columns:Expression;Reading;Meaning;Image_URI;Audio
#notetype:Japanese Anki Maker
#deck:{}

",
                deck
            )
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

    pub fn from_img(
        ocr: &MangaOcr,
        fg: &Furigana,
        translator: &mut Translator,
        img_path: &PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // let test_text = "お前わもう死んでいる";
        let img = image::open(img_path)?;

        let text = ocr.recognize(&img)?;
        let furigana = fg.to_hiragana(&text);
        // let text = "私は猫が好きです。";

        let translation = translator.translate(&text)?;

        //println!(" source: {},\n reading: {},\n translation: {}", &text, furigana,translation);

        let file_name = img_path.file_name().unwrap().to_str().unwrap().to_owned();
        Ok(Note {
            japanese: text,
            furigana: Some(furigana),
            english: translation.clone(),
            image: Some(file_name),
            audio: None,
        })
    }

    pub fn from_img_vec(
        ocr: &MangaOcr,
        fg: &Furigana,
        translator: &mut Translator,
        imgs: &[PathBuf],
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut notes = vec![];
        let mut errors = vec![];
        let mut pb = ProgressBar::new(imgs.len() as u64);
        let size = imgs.len();
        for i in 0..size {
            let img = imgs.get(i).unwrap();
            let out = Note::from_img(ocr, fg, translator, img);
            match out {
                Ok(v) => notes.push(v),
                Err(v) => errors.push((i, v)),
            }
            pb.count();
        }
        let err_count = errors.len();
        println!("{size:} files processed: {err_count:} failed");
        if errors.len() > 0 {
            for (i, v) in errors {
                let img = imgs.get(i).unwrap();
                println!("error on img {i:}: {img:?}\n\t{v:?}");
            }
        }
        Ok(notes)
    }
}
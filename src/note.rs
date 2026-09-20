use std::io::prelude::*;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use furigana::Furigana;
use manga_ocr_rs::MangaOcr;
use rayon::ThreadPoolBuilder;

use crate::progress_bar::ProgressBar;
use crate::translator::Translator;



#[derive(Clone,Debug)]
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
        // write Anki headers
        let deck = match deck_name {
            Some(v) => v,
            None => "Japanese Anki Maker Template".into(),
        };
        let mut output = PathBuf::from("_jam_output");
        std::fs::create_dir_all(&output)?;
        output.push(deck.clone()+".txt");
        let mut file = File::create(output)?;
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
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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
        translators: &[Mutex<Translator>],
        imgs: &[PathBuf],
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        
        let worker_count = translators.len();
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()?;
        
        let size = imgs.len();
        let pb = Arc::new(ProgressBar::new(size as u64));

        let results:Vec<Result<Note,Box<dyn std::error::Error + Send + Sync>>> = pool.install(||{
            imgs.par_iter()
            .map(|img| {
                let index = rayon::current_thread_index().unwrap();
                let mut translator = translators[index].lock().unwrap();
                let out = Note::from_img(ocr,fg,&mut translator,img);
                pb.count()?;
                out
            })
            .collect::<Vec<_>>()
        });

        let err_count = results.iter().fold(0,|acc,v| acc + if v.is_err() {1} else {0});
        println!("{size:} files processed: {err_count:} failed                       "); // whitespace erases progress_Bar
        let mut notes = vec![];
        results.iter()
        .enumerate()
        .for_each(|(i,v)|{
                if let Err(value) = v{
                    let img = imgs.get(i).unwrap();
                    println!("error on img {i:}: {img:?}\n\t{value:?}");
                }else if let Ok(value) = v{
                    notes.push(value.clone())
                }
            }
        );
        Ok(notes)
    }
}

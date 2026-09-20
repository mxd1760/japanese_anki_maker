use std::{io::prelude::*, sync::atomic::{AtomicU64, Ordering}};

pub struct ProgressBar {
    max: u64,
    value: AtomicU64,
    //color
    length: u8,
}

impl ProgressBar {
    pub fn new(size: u64) -> Self {
        println!();
        Self {
            max: size,
            value: AtomicU64::new(0),
            length: 20,
        }
    }
    pub fn count(&self) -> std::io::Result<()> {
        self.value.fetch_add(1, Ordering::Relaxed);
        if self.value.load(Ordering::Relaxed) > self.max {
            self.value.store(self.max,Ordering::Relaxed);
        }
        self.draw()?;
        Ok(())
    }
    pub fn draw(&self) -> std::io::Result<()>{
        let mut out = "".to_owned();
        let ratio = self.value.load(Ordering::Relaxed) as f64 / self.max as f64;
        for i in 0..self.length {
            if ratio > (i as f64 / self.length as f64) {
                out.push('#');
            } else {
                out.push('_');
            }
        }
        print!("  Progress |{}|\r", out);
        std::io::stdout().flush()?;
        Ok(())
    }
}

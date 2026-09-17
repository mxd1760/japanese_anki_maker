pub struct ProgressBar {
    max: u64,
    value: u64,
    //color
    length: u8,
}

impl ProgressBar {
    pub fn new(size: u64) -> Self {
        println!();
        Self {
            max: size,
            value: 0,
            length: 20,
        }
    }
    pub fn count(&mut self) {
        self.value += 1;
        if self.value > self.max {
            self.value = self.max
        }
        self.draw();
    }
    pub fn draw(&self) {
        let mut out = "".to_owned();
        let ratio = self.value as f64 / self.max as f64;
        for i in 0..self.length {
            if ratio > (i as f64 / self.length as f64) {
                out.push('#');
            } else {
                out.push('_');
            }
        }
        println!("Progress |{}|", out)
    }
}

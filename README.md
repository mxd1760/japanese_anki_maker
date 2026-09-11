# Japanese Anki Maker

A Rust tool for turning Japanese text from images into Anki flashcards.

The project combines Japanese OCR, furigana generation, and local machine translation to create Anki-compatible flashcards from screenshots or other images containing Japanese text.

## Pipeline

```text
Japanese image
      ↓
  Manga OCR
      ↓
Japanese text
      ↓
 Furigana
      ↓
English translation
      ↓
Anki import file
```

Each generated card can contain:

* Japanese expression
* Hiragana reading
* English translation
* Source image
* Optional audio

The current implementation exports cards as an Anki text import file with a custom note type and deck.

## Technologies

* **Rust**
* [Manga OCR](https://github.com/kha-white/manga-ocr) via `manga-ocr-rs`
* `ja-furigana` for Japanese readings
* ONNX Runtime for local inference
* SentencePiece and a FuGUMT translation model for Japanese → English translation

All OCR and translation processing is performed locally using downloaded model files.

## Current Status

This is currently a working prototype rather than a polished end-user application.

The current workflow is configured around a set of test images in `test_images/`. The next steps are to make image input and batch processing more configurable, improve Anki deck/media generation, and provide a more convenient interface for creating cards.

## Running

This project requires Rust and the model files used by the OCR and translation components.

After obtaining the required models, run:

```bash
cargo run
```

The generated Anki import file is written to:

```text
output.txt
```

The current implementation expects its translation models under:

```text
models/fugumt_onnx/
```

and test images under:

```text
test_images/
```

## Why I Made This

I wanted a way to turn Japanese text encountered in images into useful Anki cards without manually copying the text, looking up readings, translating it, and formatting each card.

The project is also an exploration of using Rust for a practical local machine-learning workflow, combining image processing, OCR, ONNX inference, text processing, and file generation in a single application.

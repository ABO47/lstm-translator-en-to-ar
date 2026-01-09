# lstm-translator-en-to-ar

A simple English-to-Arabic translator using LSTM seq2seq architecture

## Installation
```bash
pip install tensorflow numpy nltk psutil tabulate
```

## Usage
Edit the [`configs/config.py`](config.py) file to adjust dataset, model, training, and inference parameters:
```bash
class Config:
    # Data
    MAX_SAMPLES = 1000
    DATASET_PATH = 'data/dataset.txt'
    
    # Model
    EMBED_DIM = 128
    LATENT_DIM = 256

    # Training
    BATCH_SIZE = 64
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2

    # Inference
    INFERENCE_BENCHMARK_SAMPLES = 10
    BLEU_EVAL_SAMPLES = 100

    # Output paths
    OUTPUT_DIR = "outputs"
    MODEL_DIR = f"{OUTPUT_DIR}/models"
    TOKENIZER_DIR = f"{OUTPUT_DIR}/tokenizers"

    MODEL_PATH = f"{MODEL_DIR}/translation_model_word_lstm.keras"
    ENCODER_PATH = f"{MODEL_DIR}/encoder_model_word_lstm.keras"
    DECODER_PATH = f"{MODEL_DIR}/decoder_model_word_lstm.keras"
    EN_TOKENIZER_PATH = f"{TOKENIZER_DIR}/en_tokenizer.pkl"
    AR_TOKENIZER_PATH = f"{TOKENIZER_DIR}/ar_tokenizer.pkl"
```

Make sure your dataset is placed in [`data`](data) with the format:
```bash
english sentence<TAB>arabic sentence
```

Then train the model:
```bash
python -m scripts.train
```

which will generate:
```bash
translation_model_word_lstm.keras
encoder_model_word_lstm.keras
decoder_model_word_lstm.keras
en_tokenizer.pkl
ar_tokenizer.pkl
```

you can also run [`scripts/inference.py`](inference.py) to start translating without training the model again using:
```bash
python -m scripts.inference
```

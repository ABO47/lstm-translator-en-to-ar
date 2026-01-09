import os
import pickle

def ensure_dirs(config):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.TOKENIZER_DIR, exist_ok=True)

def save_models(model, encoder_model, decoder_model, en_tokenizer, ar_tokenizer, config):
    ensure_dirs(config)
    print("\nSaving...")
    model.save(config.MODEL_PATH)
    encoder_model.save(config.ENCODER_PATH)
    decoder_model.save(config.DECODER_PATH)
    with open(config.EN_TOKENIZER_PATH, "wb") as f:
        pickle.dump(en_tokenizer, f)
    with open(config.AR_TOKENIZER_PATH, "wb") as f:
        pickle.dump(ar_tokenizer, f)
    print(f"Models and tokenizers saved to:")
    print(f"  - {config.MODEL_PATH}")
    print(f"  - {config.ENCODER_PATH}")
    print(f"  - {config.DECODER_PATH}")
    print(f"  - {config.EN_TOKENIZER_PATH}")
    print(f"  - {config.AR_TOKENIZER_PATH}")

def load_tokenizers(config):
    with open(config.EN_TOKENIZER_PATH, "rb") as f:
        en_tokenizer = pickle.load(f)
    with open(config.AR_TOKENIZER_PATH, "rb") as f:
        ar_tokenizer = pickle.load(f)
    return en_tokenizer, ar_tokenizer
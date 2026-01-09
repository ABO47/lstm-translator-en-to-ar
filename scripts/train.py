import numpy as np
import time
from configs.config import Config
from data.data_utils import load_dataset, prepare_tokenizers, prepare_sequences
from models.model_builder import build_training_model, build_inference_models
from models.translator import Translator
from metrics.metrics import MetricsTracker
from utils.model_io import save_models

def train_model():
    config = Config()
    
    en_texts, ar_texts = load_dataset(config.DATASET_PATH, config.MAX_SAMPLES)
    en_tokenizer, ar_tokenizer = prepare_tokenizers(en_texts, ar_texts)
    
    data = prepare_sequences(en_texts, ar_texts, en_tokenizer, ar_tokenizer)
    
    print(f"\nDataset Info:")
    print(f"  English vocab size: {data['en_vocab_size']}")
    print(f"  Arabic vocab size: {data['ar_vocab_size']}")
    print(f"  Max English length: {data['max_en_len']}")
    print(f"  Max Arabic length: {data['max_ar_len']}")
    
    # Build model
    print("\nBuilding model...")
    model = build_training_model(
        data['en_vocab_size'],
        data['ar_vocab_size'],
        data['max_en_len'],
        config.EMBED_DIM,
        config.LATENT_DIM
    )
    
    print(model.summary())
    
    # Train model
    print(f"\nTraining model for {config.EPOCHS} epochs...")
    start_time = time.time()
    
    history = model.fit(
        [data['en_data'], data['ar_data'][:, :-1]],
        np.expand_dims(data['ar_data'][:, 1:], -1),
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT
    )
    
    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.2f} seconds")
    
    # Build inference models
    print("\nBuilding inference models...")
    encoder_model, decoder_model = build_inference_models(model, config.LATENT_DIM)
    
    # Create translator
    translator = Translator(
        encoder_model,
        decoder_model,
        en_tokenizer,
        ar_tokenizer,
        data['max_en_len'],
        data['max_ar_len']
    )
    
    # Show sample translations
    print("\nSAMPLE TRANSLATIONS:")
    for i in range(min(5, len(en_texts))):
        print(f"\nEN: {en_texts[i]}")
        print(f"AR: {ar_texts[i]}")
        print(f"PR: {translator.translate(en_texts[i])}")
        print("-" * 50)
    
    # Calculate and display metrics
    print("\nCalculating metrics...")
    metrics_tracker = MetricsTracker()
    metrics_tracker.track_training_metrics(history, train_duration)
    metrics_tracker.calculate_bleu_score(
        translator, en_texts, ar_texts, config.BLEU_EVAL_SAMPLES
    )
    metrics_tracker.track_inference_metrics(
        translator, en_texts, config.INFERENCE_BENCHMARK_SAMPLES
    )
    metrics_tracker.display_metrics()
    
    # Save models
    save_models(model, encoder_model, decoder_model, en_tokenizer, 
                ar_tokenizer, config)
    
    return translator


def interactive_mode(translator):
    """Run interactive translation mode."""
    print("\n=== INTERACTIVE MODE ===")
    print("Model ready! Type English text (or 'quit' to exit)")
    
    while True:
        try:
            text = input("> ")
            if text.lower() == "quit":
                break
            translation = translator.translate(text)
            print(f"AR: {translation}")
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    translator = train_model()
    interactive_mode(translator)
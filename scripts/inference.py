from tensorflow.keras.models import load_model
from configs.config import Config
from utils.model_io import load_tokenizers
from models.translator import Translator


def load_trained_model():

    config = Config()
    
    print("Loading models...")
    encoder_model = load_model(config.ENCODER_PATH)
    decoder_model = load_model(config.DECODER_PATH)
    
    print("Loading tokenizers...")
    en_tokenizer, ar_tokenizer = load_tokenizers(config)
    
    # Get sequence lengths from tokenizers
    max_en_len = max(len(seq) for seq in 
                     en_tokenizer.texts_to_sequences(["test"]))
    max_ar_len = max(len(seq) for seq in 
                     ar_tokenizer.texts_to_sequences(["<sos> test <eos>"]))
    
    translator = Translator(
        encoder_model,
        decoder_model,
        en_tokenizer,
        ar_tokenizer,
        max_en_len,
        max_ar_len
    )
    
    print("Models loaded successfully!")
    return translator


def main():
    translator = load_trained_model()
    
    print("\n=== TRANSLATION MODE ===")
    print("Type English text to translate (or type 'quit, exit, q' to exit)")
    
    while True:
        try:
            text = input("\nEN: ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            translation = translator.translate(text)
            print(f"AR: {translation}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
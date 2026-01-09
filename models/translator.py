import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data.data_utils import normalize_en


class Translator:
    
    def __init__(self, encoder_model, decoder_model, en_tokenizer, ar_tokenizer, max_en_len, max_ar_len):
        
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.en_tokenizer = en_tokenizer
        self.ar_tokenizer = ar_tokenizer
        self.max_en_len = max_en_len
        self.max_ar_len = max_ar_len
    
    def translate(self, text):

        text = normalize_en(text)
        seq = self.en_tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=self.max_en_len, padding="post")
        
        states = self.encoder_model.predict(seq, verbose=0)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.ar_tokenizer.word_index["<sos>"]
        
        decoded_words = []
        for _ in range(self.max_ar_len):
            output, h, c = self.decoder_model.predict([target_seq] + states, 
                                                      verbose=0)
            idx = np.argmax(output[0, -1, :])
            word = self.ar_tokenizer.index_word.get(idx, "")
            
            if word == "<eos>":
                break
            
            decoded_words.append(word)
            target_seq[0, 0] = idx
            states = [h, c]
        
        return " ".join(decoded_words)
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def normalize_en(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_ar(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(file_path, max_samples=None):
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    
    en_texts = []
    ar_texts = []
    
    print(f"Loading dataset from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if '\t' in line:
                en, ar = line.split('\t')[:2]
                en_texts.append(normalize_en(en))
                ar_texts.append(normalize_ar(ar))
    
    print(f"Loaded {len(en_texts)} translation pairs")
    return en_texts, ar_texts


def prepare_tokenizers(en_texts, ar_texts):
    en_tokenizer = Tokenizer(oov_token="<unk>", filters="")
    ar_tokenizer = Tokenizer(oov_token="<unk>", filters="")
    
    en_tokenizer.fit_on_texts(en_texts)
    ar_tokenizer.fit_on_texts(["<sos> " + t + " <eos>" for t in ar_texts])
    
    return en_tokenizer, ar_tokenizer


def prepare_sequences(en_texts, ar_texts, en_tokenizer, ar_tokenizer):
    en_seq = en_tokenizer.texts_to_sequences(en_texts)
    ar_seq = ar_tokenizer.texts_to_sequences(["<sos> " + t + " <eos>" for t in ar_texts])
    
    max_en_len = max(len(s) for s in en_seq)
    max_ar_len = max(len(s) for s in ar_seq)
    
    en_data = pad_sequences(en_seq, maxlen=max_en_len, padding="post")
    ar_data = pad_sequences(ar_seq, maxlen=max_ar_len, padding="post")
    
    en_vocab_size = len(en_tokenizer.word_index) + 1
    ar_vocab_size = len(ar_tokenizer.word_index) + 1
    
    return {
        'en_data': en_data,
        'ar_data': ar_data,
        'max_en_len': max_en_len,
        'max_ar_len': max_ar_len,
        'en_vocab_size': en_vocab_size,
        'ar_vocab_size': ar_vocab_size
    }
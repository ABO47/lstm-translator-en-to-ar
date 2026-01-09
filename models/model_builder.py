from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding


def build_training_model(en_vocab_size, ar_vocab_size, max_en_len, embed_dim, latent_dim):
    
    # Encoder
    encoder_inputs = Input(shape=(max_en_len,), name='encoder_input')
    enc_emb = Embedding(en_vocab_size, embed_dim, mask_zero=True, 
                        name='encoder_embedding')(encoder_inputs)
    encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, h, c = encoder_lstm(enc_emb)
    encoder_states = [h, c]
    
    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_input')
    dec_emb = Embedding(ar_vocab_size, embed_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(ar_vocab_size, activation="softmax", name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Defining the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    return model


def build_inference_models(training_model, latent_dim):

    # Encoder model
    encoder_inputs = training_model.input[0]
    encoder_lstm = training_model.get_layer('encoder_lstm')
    encoder_outputs, h, c = encoder_lstm.output
    encoder_states = [h, c]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Decoder model
    decoder_inputs = training_model.input[1]
    dec_state_h = Input(shape=(latent_dim,), name='dec_state_h')
    dec_state_c = Input(shape=(latent_dim,), name='dec_state_c')
    dec_states_inputs = [dec_state_h, dec_state_c]
    
    dec_emb_layer = training_model.get_layer('decoder_embedding')
    decoder_lstm = training_model.get_layer('decoder_lstm')
    decoder_dense = training_model.get_layer('decoder_dense')
    
    dec_emb2 = dec_emb_layer(decoder_inputs)
    dec_outputs2, h2, c2 = decoder_lstm(dec_emb2, initial_state=dec_states_inputs)
    dec_outputs2 = decoder_dense(dec_outputs2)
    
    decoder_model = Model([decoder_inputs] + dec_states_inputs, [dec_outputs2, h2, c2])
    
    return encoder_model, decoder_model
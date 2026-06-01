import os
import keras
from ML_Model_Builder import layers as my_layers

class Decoder_Builder:
    def __init__(self, model_name, num_decoder_blocks, num_heads, key_dims, maxlen, vocab_size, embed_dim):
        self.model_name = model_name
        self.num_decoder_blocks = num_decoder_blocks
        self.num_heads = num_heads
        self.key_dims = key_dims
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
    def build_decoder(self, query = None):
        inp = keras.layers.Input(shape=(self.maxlen, ))
        pos_emd = my_layers.token_and_position_embedding(maxlen=self.maxlen, vocab_size=self.vocab_size, embed_dim=self.embed_dim)(inp)
        x = pos_emd
        for _ in range(self.num_decoder_blocks):
            x = my_layers.decoder_block(num_heads=self.num_heads, key_dims=self.key_dims)(values = x, queries = x if query is None else query)
            query = x
        x = keras.layers.Dense(units=self.vocab_size, activation='softmax')(x)
        model = keras.Model(inputs=inp, outputs=x, name=self.model_name)
        self.model = model
        return model
    
    def get_model(self):
        return self.model
    
    def summary(self):
        self.model.summary()
    
    def train_decoder(self, x_train, y_train, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def save_model(self, file_path):
        self.model.save(file_path)
        
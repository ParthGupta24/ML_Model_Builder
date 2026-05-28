import os
import keras
<<<<<<< HEAD
import layers
=======
from keras import ops

class token_and_position_embedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(token_and_position_embedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(maxlen)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.pos_emb.input_dim,
            'vocab_size': self.token_emb.input_dim,
            'embed_dim': self.token_emb.output_dim,
        })
        return config
    
    def from_config(cls, config):
        return cls(**config)

class encoder_block(keras.layers.Layer):
    def __init__(self, num_heads, key_dims):
        super(encoder_block, self).__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dims)
        self.ffn = keras.Sequential([
            keras.layers.Dense(4 * key_dims, activation='relu'),
            keras.layers.Dense(key_dims)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.mha(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.mha.num_heads,
            'key_dims': self.mha.key_dim,
        })
        return config
    
    def from_config(cls, config):
        return cls(**config)
>>>>>>> c069a1c8cc566550bc92d03a686758f1edfa37c5

class Encoder_Builder:
    def __init__(self, model_name, num_encoder_blocks, num_heads, key_dims, maxlen, vocab_size, embed_dim):
        self.model_name = model_name
        self.num_encoder_blocks = num_encoder_blocks
        self.num_heads = num_heads
        self.key_dims = key_dims
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def build_encoder(self):
        inp = keras.layers.Input(shape=(self.maxlen, ))
        pos_emd = layers.token_and_position_embedding(maxlen=self.maxlen, vocab_size=self.vocab_size, embed_dim=self.embed_dim)(inp)
        x = pos_emd
        for _ in range(self.num_encoder_blocks):
            x = layers.encoder_block(num_heads=self.num_heads, key_dims=self.key_dims)(x)
        model = keras.Model(inputs=inp, outputs=x, name=self.model_name)
        self.model = model
        return model
    
    def get_model(self):
        return self.model
    
    def summary(self):
        self.model.summary()
        
    def train_encoder(self, x_train, y_train, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def save_model(self, file_path):
        self.model.save(file_path)
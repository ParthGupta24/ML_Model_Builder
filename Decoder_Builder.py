import keras
import tensorflow as tf
        
class token_and_position_embedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(token_and_position_embedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
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


class decoder_block(keras.layers.Layer):
    def __init__(self, num_heads, key_dims):
        super(decoder_block, self).__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dims)
        self.ffn = keras.Sequential([
            keras.layers.Dense(4 * key_dims, activation='relu'),
            keras.layers.Dense(key_dims)
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, values, queries):
        attn_output = self.mha(values, queries, use_causal_mask=True)
        out1 = self.layernorm1(values + attn_output)
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
        pos_emd = token_and_position_embedding(maxlen=self.maxlen, vocab_size=self.vocab_size, embed_dim=self.embed_dim)(inp)
        x = pos_emd
        for _ in range(self.num_decoder_blocks):
            x = decoder_block(num_heads=self.num_heads, key_dims=self.key_dims)(values = x, queries = x if query is None else query)
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
        
import keras
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

class patch_embedding(keras.layers.Layer):
    def __init__(self, maxlen, patch_size, embed_dim):
        super(patch_embedding, self).__init__()
        self.patch_emb = keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='same')
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        patches = self.patch_emb(inputs)
        patches = keras.layers.Reshape((patches.shape[1] * patches.shape[2], patches.shape[3]))(patches)
        positions = ops.arange(ops.shape(patches)[1])
        pos_embeddings = self.pos_emb(positions)
        return patches + pos_embeddings
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_emb.kernel_size,
            'vocab_size': self.pos_emb.input_dim,
            'embed_dim': self.pos_emb.output_dim
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
import keras

class UNet_Builder:
    def __init__(self, model_name, input_shape, num_classes, filters=[64, 128, 256, 512]):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
    
    def encoder_block(self, inputs, filters):
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
        p = keras.layers.MaxPooling2D((2, 2))(c)
        return c, p
    
    def decoder_block(self, inputs, skip_features, filters):
        u = keras.layers.UpSampling2D((2, 2))(inputs)
        u = keras.layers.concatenate([u, skip_features])
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(u)
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
        return c
    
    def bottleneck(self, inputs, filters):
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        c = keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(c)
        return c
        
    def build_unet(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        skip_connections = []
        x = inputs
        for f in self.filters:
            s, x = self.encoder_block(x, f)
            skip_connections.append(s)
        x = self.bottleneck(x, self.filters[-1] * 2)
        skip_connections = skip_connections[::-1]
        for i, f in enumerate(self.filters[::-1]):
            x = self.decoder_block(x, skip_connections[i], f)
        outputs = keras.layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_name)
        return self.model
    
    def get_model(self):
        return self.model
    
    def summary(self):
        self.model.summary()
        
    def train_Unet(self, x_train, y_train, batch_size, epochs):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    
    def save_model(self, file_path):
        self.model.save(file_path)
        
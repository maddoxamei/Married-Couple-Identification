import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dimension, hidden_layers = 1, log_progression = False, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
              units=latent_dimension,
              activation=tf.nn.relu,
              kernel_initializer='he_uniform'
        )
        self.output_layer = tf.keras.layers.Dense(
              units=latent_dimension,
              activation=tf.nn.sigmoid
        )
    
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dimension, hidden_layers = 1, log_progression = False, **kwargs):
        super(Decoder, self).__init__()
        self.output_dimension = output_dimension
        self.output_layer = tf.keras.layers.Dense(
              units=output_dimension,
              activation=tf.nn.sigmoid
        )
        
    def build(self, input_shape):
        self.hidden_layer = tf.keras.layers.Dense(
              units=input_shape[-1],
              activation=tf.nn.relu,
              kernel_initializer='he_uniform'
        )
    
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)
    
class Autoencoder(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(*args)
        self.output_dimension = kwargs.get('output_dimension')
        
    def build(self, input_shape):
        if self.output_dimension is None:
            self.decoder = Decoder(input_shape[-1])
        else:
            self.decoder = Decoder(self.output_dimension)
        
    def call(self, input_features):
        encoded = self.encoder(input_features)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == "__main__":
    
    """
    # Works for 3-D tensors (e.g. collection of images) too
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train.shape, x_test.shape
    """
    
    x_train = np.random.rand(100,8)
    x_test = np.random.rand(15,8)

    print( x_train.shape, x_test.shape )
    
    autoencoder = Autoencoder(4)
    autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())
    history = autoencoder.fit(x_train, x_train,
                    epochs=1,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    autoencoder.summary()
    print(history.history)
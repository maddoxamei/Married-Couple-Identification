import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dimension, hidden_layers = 1, log_progression = False, activation = None, **kwargs):
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.log_progression = log_progression
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        if self.log_progression:
            self.units = np.geomspace(input_shape[-1], self.latent_dimension, self.hidden_layers + 1)[1:]
        else:
            self.units = np.linspace(input_shape[-1], self.latent_dimension, self.hidden_layers + 1)[1:]
        self.hidden_layers = [tf.keras.layers.Dense(units = units, activation = self.activation) for units in self.units]
    
    def call(self, x): # x represents the input features/tensor
        for layer in self.hidden_layers:
            x = layer(x)
        return x
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_dimension, hidden_layers = 1, log_progression = False, activation = None, **kwargs):
        super(Decoder, self).__init__()
        self.output_dimension = output_dimension
        self.hidden_layers = hidden_layers
        self.log_progression = log_progression
        self.activation = tf.keras.activations.get(activation)
        self.output_layer = tf.keras.layers.Dense(
              units=output_dimension,
              activation=tf.nn.sigmoid
        )
        
    def build(self, input_shape):
        if self.log_progression:
            self.units = np.geomspace(input_shape[-1], self.output_dimension, self.hidden_layers + 1)[1:-1]
        else:
            self.units = np.linspace(input_shape[-1], self.output_dimension, self.hidden_layers + 1)[1:-1]
        self.hidden_layers = [tf.keras.layers.Dense(units = units, activation = self.activation) for units in self.units]
    
    def call(self, x): # x represents the input features/tensor    
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
class Autoencoder(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.kwargs = kwargs
        self.output_dimension = kwargs.get('output_dimension')
        
    def build(self, input_shape):
        if self.output_dimension is None:
            self.decoder = Decoder(input_shape[-1], **self.kwargs)
        else:
            self.kwargs.pop('output_dimension')
            self.decoder = Decoder(self.output_dimension, **self.kwargs)
        
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
    
    autoencoder = Autoencoder(4, hidden_layers = 2, log_progression = False, activation = tf.nn.relu)
    autoencoder.compile(optimizer='adam', loss=tf.losses.MeanSquaredError())
    history = autoencoder.fit(x_train, x_train,
                    epochs=1,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    
    autoencoder.summary()
    print(history.history)
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

class BetaVAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim=16, beta=4.0):
        super(BetaVAE, self).__init__()
        self.input_dim = input_dim
        self.beta = beta
        # Encoder: Compresses lyrics into latent space
        self.encoder = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim * 2) 
        ])
        # Decoder: Reconstructs lyrics from latent space
        self.decoder = models.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def _sampling(self, z_mean, z_log_var):
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        params = self.encoder(inputs)
        mean, log_var = tf.split(params, 2, axis=1)
        z = self._sampling(mean, log_var)
        recon = self.decoder(z)
        
        # VAE Loss = Reconstruction Loss + Beta * KL Divergence
        recon_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, recon)) * self.input_dim
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
        self.add_loss(recon_loss + (self.beta * kl_loss))
        return recon

    def get_latent(self, inputs):
        params = self.encoder(inputs)
        mean, log_var = tf.split(params, 2, axis=1)
        return self._sampling(mean, log_var)

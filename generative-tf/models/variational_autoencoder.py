import tensorflow as tf
import numpy as np

class VariationalAutoencoder():
    def __init__(self, 
                input_dim, 
                latent_dim, 
                hidden_dim=10, 
                batch_size=100, 
                num_layers=0):

        self.graph = tf.Graph()

        with self.graph.as_default():
            ## Input x variable
            self.x = tf.placeholder(tf.float32, shape=(None, input_dim))

            def init_uniform_scaled(in_dim, out_dim):
                extreme = np.sqrt(6.0 / (in_dim * out_dim))
                return tf.random_uniform(
                    (in_dim, out_dim), minval=-extreme, maxval=extreme, dtype=tf.float32)

            self._latent_dim = latent_dim
            self.batch_size = batch_size

            self._encoder_W = tf.Variable(init_uniform_scaled(input_dim, hidden_dim))
            self._encoder_bias = tf.Variable(tf.zeros([hidden_dim]))

            self._mean_encoder = tf.Variable(init_uniform_scaled(hidden_dim, latent_dim))
            self._mean_encoder_bias = tf.Variable(tf.zeros([latent_dim]))

            self._log_stddev_encoder = tf.Variable(init_uniform_scaled(hidden_dim, latent_dim))
            self._log_stddev_encoder_bias = tf.Variable(tf.zeros([latent_dim]))

            self._decoder_W = tf.Variable(init_uniform_scaled(latent_dim, hidden_dim))
            self._decoder_bias = tf.Variable(tf.zeros([hidden_dim]))

            self._mean_decoder = tf.Variable(init_uniform_scaled(hidden_dim, input_dim))
            self._mean_decoder_bias = tf.Variable(tf.zeros([input_dim]))

    def _generate(self, z):    
        with self.graph.as_default():   

            # Compute the hidden state from latent variables  
            h = tf.nn.softplus(
                    tf.matmul(z, self._decoder_W) + self._decoder_bias
                )

            # Compute the reconstruction from hidden state
            mean = tf.nn.sigmoid(
                    tf.matmul(h, self._mean_decoder) + self._mean_decoder_bias
                )

            log_stddev = None
            # log_stddev = tf.nn.relu(
            #         tf.matmul(h, self._log_stddev_decoder) + self._log_stddev_decoder_bias
            #     )

        return (mean, log_stddev)

    def _encode(self):
        """
          Forward step of the variational autoencoder

          Takes input

        """
        with self.graph.as_default():
            h = tf.nn.relu(
                    tf.matmul(self.x, self._encoder_W) + self._encoder_bias
                )

            latent_mean = tf.nn.relu(
                    tf.matmul(h, self._mean_encoder) + self._mean_encoder_bias
                )

            latent_log_stddev = tf.nn.relu(
                    tf.matmul(h, self._log_stddev_encoder) + self._log_stddev_encoder_bias
                )

        return (latent_mean, latent_log_stddev)

    def _loss(self, tol=1e-4):
        """
            Variational objective function

            ELBO = log joint log-likelihood(p, q) + log q

        """
        with self.graph.as_default():

            # Forward pass of data into latent space
            mean_encoder, log_stddev_encoder = self._encode()

            random_noise = tf.random_normal(
                (self.batch_size, self._latent_dim), 0, 1, dtype=tf.float32)

            # Reparameterization trick of re-scaling/transforming random error
            z = mean_encoder + tf.exp(log_stddev_encoder) * random_noise            

            # Reconstruction/decoing of latent space
            mean_decoder, _ = self._generate(z)

            # Bernoulli log-likelihood reconstruction
            # TODO: other distributon types
            reconstruction_error = tf.reduce_sum(
                    (self.x * tf.log(tol + mean_decoder)) 
                        + ((1 - self.x) * tf.log(tol+ 1 - mean_decoder)), 1)

            # Gaussian entropy
            # Optimizing this smoothes out the variatonal distribution - occupying more states
            entropy = \
                -0.5 * tf.reduce_sum(1 
                                + log_stddev_encoder 
                                - tf.square(mean_encoder) 
                                - tf.exp(log_stddev_encoder), 1) 

        return tf.reduce_mean(reconstruction_error + entropy)
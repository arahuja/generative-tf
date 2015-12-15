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

            self._log_variance_encoder = tf.Variable(init_uniform_scaled(hidden_dim, latent_dim))
            self._log_variance_encoder_bias = tf.Variable(tf.zeros([latent_dim]))

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

            latent_log_variance = tf.nn.relu(
                    tf.matmul(h, self._log_variance_encoder) + self._log_variance_encoder_bias
                )

        return (latent_mean, latent_log_variance)

    def _evidence_lower_bound(self, importance_weighting=False, tol=1e-4):
        """
            Variational objective function

            ELBO = E(log joint log-likelihood) + E(log q)
                 = MC estimate of log joint + Entropy(q)

        """
        with self.graph.as_default():

            # Forward pass of data into latent space
            mean_encoder, log_variance_encoder = self._encode()

            random_noise = tf.random_normal(
                (self.batch_size, self._latent_dim), 0, 1, dtype=tf.float32)

            # Reparameterization trick of re-scaling/transforming random error
            std_dev = tf.sqrt(tf.exp(log_variance_encoder))
            z = mean_encoder + std_dev * random_noise            

            # Reconstruction/decoing of latent space
            mean_decoder, _ = self._generate(z)

            # Bernoulli log-likelihood reconstruction
            # TODO: other distributon types

            def bernoulli_log_joint(x):
                return tf.reduce_sum(
                    (x * tf.log(tol + mean_decoder))
                        + ((1 - x) * tf.log(tol + 1 - mean_decoder)), 
                    1)

            p_log_joint = bernoulli_log_joint(self.x)

            # Gaussian entropy
            # Optimizing this smoothes out the variatonal distribution - occupying more states
            entropy = \
                -0.5 * tf.reduce_sum(1 
                                + log_variance_encoder
                                - tf.square(mean_encoder) 
                                - tf.exp(log_variance_encoder), 1) 

            if importance_weighting:
                log_weights = (p_log_joint - entropy) 
                log_weights_scaled = log_weights - log_weights.max()
                weights = tf.exp(log_weights_scaled)
                weights_normalized = weights / weights.sum()
                objective = tf.dot(log_weights, weights_normalized)
            else:
                objective = tf.reduce_mean(p_log_joint - entropy)

        return objective
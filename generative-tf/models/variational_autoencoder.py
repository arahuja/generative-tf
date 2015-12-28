import tensorflow as tf
import numpy as np
from initialization import xavier_glorot_initialization

class VariationalAutoencoder():
    def __init__(self, 
                input_dim, 
                latent_dim, 
                hidden_dim=10, 
                batch_size=100, 
                num_layers=0,
                activation_func=tf.nn.relu,
                output_activation_func=tf.nn.sigmoid):

        self.graph = tf.Graph()
        self.activation_func = activation_func
        self.output_activation_func = output_activation_func
        self.input_dim = input_dim
        self.batch_size = batch_size

        with self.graph.as_default():
            ## Input x variable
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, input_dim))
         
            ## Dimension of the latent variables mu/mean and log_variance
            self._latent_dim = latent_dim
            self.batch_size = batch_size

            self._encoder_W = tf.Variable(xavier_glorot_initialization(input_dim, hidden_dim))
            self._encoder_bias = tf.Variable(tf.zeros([hidden_dim]))

            self._mean_encoder = tf.Variable(xavier_glorot_initialization(hidden_dim, latent_dim))
            self._mean_encoder_bias = tf.Variable(tf.zeros([latent_dim]))

            self._log_variance_encoder = tf.Variable(xavier_glorot_initialization(hidden_dim, latent_dim))
            self._log_variance_encoder_bias = tf.Variable(tf.zeros([latent_dim]))

            self._decoder_W = tf.Variable(xavier_glorot_initialization(latent_dim, hidden_dim))
            self._decoder_bias = tf.Variable(tf.zeros([hidden_dim]))

            self._mean_decoder = tf.Variable(xavier_glorot_initialization(hidden_dim, input_dim))
            self._mean_decoder_bias = tf.Variable(tf.zeros([input_dim]))

    def _generate(self,
                 z,
                 activation_func=tf.nn.softplus,
                 output_activation_func=tf.nn.sigmoid):    
        with self.graph.as_default():   

            # Compute the hidden state from latent variables  
            h = activation_func(
                    tf.matmul(z, self._decoder_W) + self._decoder_bias
                )

            # Compute the reconstruction from hidden state
            mean = output_activation_func(
                    tf.matmul(h, self._mean_decoder) + self._mean_decoder_bias
                )

            log_variance = None

        return (mean, log_variance)

    def _encode(self, x):
        """
          Forward step of the variational autoencoder

          Takes input

        """
        with self.graph.as_default():
            h = self.activation_func(
                    tf.matmul(x, self._encoder_W) + self._encoder_bias
                )

            latent_mean = self.activation_func(
                    tf.matmul(h, self._mean_encoder) + self._mean_encoder_bias
                )

            latent_log_variance = self.activation_func(
                    tf.matmul(h, self._log_variance_encoder) + self._log_variance_encoder_bias
                )

        return (latent_mean, latent_log_variance)

    def _evidence_lower_bound(self,
                              monte_carlo_samples=5,
                              importance_weighting=False,
                              tol=1e-5):
        """
            Variational objective function

            ELBO = E(log joint log-likelihood) - E(log q)
                 = MC estimate of log joint - Entropy(q)

        """

        with self.graph.as_default():
            x_resampled = tf.tile(self.x, tf.constant([monte_carlo_samples, 1]))

            # Forward pass of data into latent space
            mean_encoder, log_variance_encoder = self._encode(x_resampled)

            random_noise = tf.random_normal(
                (self.batch_size * monte_carlo_samples, self._latent_dim), 0, 1, dtype=tf.float32)

            # Reparameterization trick of re-scaling/transforming random error
            std_dev = tf.sqrt(tf.exp(log_variance_encoder))
            z = mean_encoder + std_dev * random_noise

            # Reconstruction/decoding of latent space
            mean_decoder, _ = self._generate(z)

            # Bernoulli log-likelihood reconstruction
            # TODO: other distributon types
            def bernoulli_log_joint(x):
                return tf.reduce_sum(
                    (x * tf.log(tol + mean_decoder))
                        + ((1 - x) * tf.log(tol + 1 - mean_decoder)), 
                    1)

            log2pi = tf.log(2.0 * np.pi)

            def gaussian_likelihood(data, mean, log_variance):
                """Log-likelihood of data given ~ N(mean, exp(log_variance))
                
                Parameters
                ----------
                data : 
                    Samples from Gaussian centered at mean
                mean : 
                    Mean of the Gaussian distribution
                log_variance : 
                    Log variance of the Gaussian distribution

                Returns
                -------
                log_likelihood : float

                """

                num_components = data.get_shape().as_list()[1]
                variance = tf.exp(log_variance)
                log_likelihood = (
                    -(log2pi * (num_components / 2.0))
                    - tf.reduce_sum(
                        (tf.square(data - mean) / (2 * variance)) + (log_variance / 2.0),
                        1)
                )

                return log_likelihood

            def standard_gaussian_likelihood(data):
                """Log-likelihood of data given ~ N(0, 1)

                Parameters
                ----------
                data : 
                    Samples from Guassian centered at 0

                Returns
                -------
                log_likelihood : float

                """

                num_components = data.get_shape().as_list()[1]
                log_likelihood = (
                    -(log2pi * (num_components / 2.0))
                    - tf.reduce_sum(tf.square(data) / 2.0, 1)
                )

                return log_likelihood

            log_p_given_z = bernoulli_log_joint(x_resampled)

            if importance_weighting:
                log_q_z = gaussian_likelihood(z, mean_encoder, log_variance_encoder)
                log_p_z = standard_gaussian_likelihood(z)

                regularization_term = log_p_z - log_q_z
            else:
                # Analytic solution to KL(q_z | p_z)
                p_z_q_z_kl_divergence = \
                    -0.5 * tf.reduce_sum(1 
                                    + log_variance_encoder
                                    - tf.square(mean_encoder) 
                                    - tf.exp(log_variance_encoder), 1) 

                regularization_term = -p_z_q_z_kl_divergence

            log_p_given_z_mc = tf.reshape(log_p_given_z, 
                                        [self.batch_size, monte_carlo_samples])
            regularization_term_mc = tf.reshape(regularization_term,
                                [self.batch_size, monte_carlo_samples])

            log_weights = log_p_given_z_mc + regularization_term_mc

            if importance_weighting:
                # Need to compute normalization constant for weights, which is
                # log (sum (exp(log_weights)))
                # weights_iw = tf.log(tf.sum(tf.exp(log_weights)))

                # Instead using log-sum-exp trick
                wmax = tf.reduce_max(log_weights, 1, keep_dims=True)

                # w_i = p_x/ q_z, log_wi = log_p_joint - log_qz
                # log ( 1/k * sum(exp(log w_i)))
                weights_iw = tf.log(tf.reduce_mean(tf.exp(log_weights - wmax), 1))
                objective = tf.reduce_mean(wmax) + tf.reduce_mean(weights_iw)
            else:
                objective = tf.reduce_mean(log_weights)

        return objective
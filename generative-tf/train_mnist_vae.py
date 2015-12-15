import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.variational_autoencoder import VariationalAutoencoder


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train_test_mnist_vae(epochs,
                         print_n, 
                         hidden_dim=10,
                         latent_dim=10,
                         batch_size=100,
                         optimizer='adam',
                        ):

    input_dim = 784

    vae = VariationalAutoencoder(
                input_dim, 
                latent_dim=latent_dim, 
                hidden_dim=hidden_dim, 
                batch_size=batch_size,
            )
    with vae.graph.as_default():
        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(1e-4).minimize(-vae._evidence_lower_bound())
        elif optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9).minimize(-vae._evidence_lower_bound())
        else:
            raise ValueError("Optimizer '{}' is not supported".format(optimizer))

        with tf.Session(graph=vae.graph) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in range(epochs):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(opt, feed_dict={vae.x: batch_xs})
                if i % print_n == 0:
                    elbo = sess.run(vae._evidence_lower_bound(), feed_dict={vae.x: batch_xs})
                    test_elbo = sess.run(vae._evidence_lower_bound(), feed_dict={vae.x: mnist.test.images[:batch_size, ::]})
             
                    print("At batch: {}; batch ELBO: {}; test batch ELBO {}".format(i, elbo, test_elbo))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--print-every-N-iter', dest='print_n', default=25, type=int)
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=10, type=int)
    parser.add_argument('--latent-dim', dest='latent_dim', default=10, type=int)
    parser.add_argument('--batch-size', dest='batch_size', default=100, type=int)

    parser.add_argument('--optimizer', dest='optimizer', default='adam', type=str)

    args = parser.parse_args()

    train_test_mnist_vae(args.epochs,
        args.print_n,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        optimizer=args.optimizer
    )
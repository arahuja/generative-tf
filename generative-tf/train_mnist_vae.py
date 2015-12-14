import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.variational_autoencoder import VariationalAutoencoder


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train_test_mnist_vae(epochs,
                        print_n, 
                        hidden_dim=10,
                        latent_dim=10,
                        batch_size=100):

    input_dim = 784

    vae = VariationalAutoencoder(
                input_dim, 
                latent_dim=latent_dim, 
                hidden_dim=hidden_dim, 
                batch_size=batch_size
            )
    with vae.graph.as_default():
        opt = tf.train.AdamOptimizer(1e-4).minimize(vae._loss())

        with tf.Session(graph=vae.graph) as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in range(epochs):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(opt, feed_dict={vae.x: batch_xs})
                if i % print_n == 0:
                    nll = sess.run(vae._loss(), feed_dict={vae.x: batch_xs})
                    print("Loss at batch {} is {}".format(i, nll))

                    test_nll = sess.run(vae._loss(), feed_dict={vae.x: mnist.test.images[:batch_size, ::]})
                    print("Test NLL at batch {} is {}".format(i, test_nll))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--print-every-N-iter', dest='print_n', default=25, type=int)
    parser.add_argument('--epochs', dest='epochs', default=1000, type=int)

    print('Hello')
    args = parser.parse_args()
    print(args)

    train_test_mnist_vae(args.epochs, args.print_n)
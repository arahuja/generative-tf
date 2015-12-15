# generative-tf
Generative Models with TensorFlow


#### Variational Autoencoder

> Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes

```python
python generative-tf/train_mnist_vae.py \
     --epochs 5000 \
     --print-every-N 100 \
     --latent-dim 10 \
     --hidden-dim 500 \
     --batch-size 100 \
     --optimizer rmsprop
```

> Burda, Y., Grosse, R. & Salakhutdinov, R. Importance Weighted Autoencoders. 1–12 (2015). at <http://arxiv.org/abs/1509.00519>

Add importance weighting with:

```python
--importance-weighting
```
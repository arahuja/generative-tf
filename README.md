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

from train import train

data_dir = '/data'
batch_size = 64
img_size = 256
latent_dim = 100
num_epochs = 100

train(data_dir, batch_size, img_size, latent_dim, num_epochs)

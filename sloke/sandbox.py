import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from compressor_compressai import Network
import torch.nn.functional as F
from sit1m_data_preprocessing import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import math

checkpoint = torch.load("compressai_losses_2.pth")

bpp_loss_arr = checkpoint["bpp_loss_arr"]
aux_loss_arr = checkpoint["aux_loss_arr"]
mse_loss_arr = checkpoint["mse_loss_arr"]

plt.plot(mse_loss_arr, label="Reconstruction MSE loss")
plt.plot(aux_loss_arr, label = "Auxiliary loss")
plt.plot(bpp_loss_arr, label = "Compression loss")
plt.legend()
plt.savefig("test_2.png")


# hyperparamters    
# seed = 44
# EPOCHS = 40
# BATCH_SIZE = 64
# lmbda = 1 # parameter for bitrate distortion tradeoff

# torch.random.manual_seed(seed)
# model = Network(64).to("cuda:1")


# download_path = "../sift1m"
# splits = build_sift1m(download_path)

# # can replace the below
# train_split = get_train_split(splits)

# D = train_split.shape[1]

# for i_epoch in range(EPOCHS):
#     dataloader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
#     for i_batch, batch in enumerate(dataloader):
#         x = batch.to("cuda:1")
#         x = x.reshape(64, 1, 128)
#         y_hat, x_hat, y_likelihoods = model(x)
#         activs = y_hat.detach().cpu().numpy().flatten()
#         plt.hist(activs, bins=15)
#         plt.savefig("compressai_activations_histogram_3.png")
#         break
#     break
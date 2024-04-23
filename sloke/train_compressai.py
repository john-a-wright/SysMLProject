import torch.optim as optim
from compressor_compressai import Network
import torch.nn.functional as F
from sit1m_data_preprocessing import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import math

"""
copied stuff from https://interdigitalinc.github.io/CompressAI/tutorials/tutorial_custom.html
"""

# hyperparamters    
seed = 44
EPOCHS = 40
BATCH_SIZE = 64
lmbda = 1 # parameter for bitrate distortion tradeoff

torch.random.manual_seed(seed)
model = Network(64).to("cuda:1")


# parameters
parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

# data
download_path = "../sift1m"
splits = build_sift1m(download_path)

# can replace the below
train_split = get_train_split(splits)

D = train_split.shape[1]


bpp_loss_arr = np.array([])
aux_loss_arr = np.array([])
mse_loss_arr = np.array([])


for i_epoch in range(EPOCHS):
    dataloader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    for i_batch, batch in enumerate(dataloader):
        x = batch.to("cuda:1")
        x = x.to(torch.float)
        x = x.reshape(BATCH_SIZE, 1, D)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        _, x_hat, y_likelihoods = model(x)

         # bitrate of the quantized latent
        N, _, L = x.size()
        num_logits = N * L
        bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_logits)

        # mean square error
        mse_loss = F.mse_loss(x, x_hat)

        # final loss term
        loss = mse_loss + lmbda * bpp_loss

        loss.backward()
        optimizer.step()
        aux_loss = model.aux_loss()
        aux_optimizer.step()
        print(f"Epoch: {i_epoch}, Batch: {i_batch}, loss: {loss}, aux_loss: {aux_loss}")
        bpp_loss_arr = np.append(bpp_loss_arr, bpp_loss.detach().cpu().numpy())
        aux_loss_arr = np.append(aux_loss_arr, aux_loss.detach().cpu().numpy())
        mse_loss_arr = np.append(mse_loss_arr, mse_loss.detach().cpu().numpy())

    torch.save({
        'epoch': i_epoch,
        'model_state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bpp_loss_arr': bpp_loss_arr,
        'aux_loss_arr': aux_loss_arr,
        'mse_loss_arr': mse_loss_arr
    }, f"compressai_losses_2.pth")

plt.plot(mse_loss_arr, legend="Reconstruction MSE loss")
plt.plot(aux_loss_arr, legend = "Auxiliary loss")
plt.plot(bpp_loss, legend = "Compression loss")
plt.legend()
plt.savefig("compressai_losses_2.png")
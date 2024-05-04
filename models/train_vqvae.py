from sit1m_data_preprocessing import *
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vqvae import VectorQuantizedAutoencoder
import matplotlib.pyplot as plt
import torch.nn as nn

"""checking how to use the sift1m data preprocessing
"""

download_path = "../sift1m"
splits = build_sift1m(download_path)

train_split = get_train_split(splits)


lr = 3e-3
seed = 44
levels =  [8, 8, 8, 5, 5, 5]
EPOCHS = 40
BATCH_SIZE = 64
D = train_split.shape[1]


torch.random.manual_seed(seed)
model = VectorQuantizedAutoencoder(levels, len(levels)).to("cuda:1")
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss = nn.MSELoss()

log_loss = np.array([])
for i_epoch in range(EPOCHS):
    dataloader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    for i_batch, batch in enumerate(dataloader):
        x = batch.to("cuda:1")
        x = x.to(torch.float)
        x = x.reshape(BATCH_SIZE, 1, D)

        optimizer.zero_grad()
        out, indices = model(x)

        rec_loss = loss(out, x)
        rec_loss.backward()
        optimizer.step()
        print(f"Epoch: {i_epoch}, Batch: {i_batch}, Loss: {rec_loss}")
        log_loss = np.append(log_loss,np.log(rec_loss.detach().cpu().numpy()))

    torch.save({
        'epoch': i_epoch,
        'model_state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'log_loss': log_loss,
    }, f"fsq_vqvae.pth")

plt.plot(log_loss, label = "log loss")
plt.legend()
plt.savefig("vqvae_fsq_loss.png")
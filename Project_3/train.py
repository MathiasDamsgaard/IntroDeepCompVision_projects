from time import time

import torch
from dataset.driveDataset import Drive
from losses import BCELoss, BCELoss_TotalVariation, DiceLoss, FocalLoss
from model.EncDecModel import EncDec
from model.UNetModel import UNet
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

# Dataset
size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

batch_size = 6
workers = 3
trainset = Drive(train=True, transform=train_transform)
train_full_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
testset = Drive(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Validation setup - split from training set
val_size = int(0.2 * len(trainset))
train_size = len(trainset) - val_size
train_subset, val_subset = torch.utils.data.random_split(trainset, [train_size, val_size])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
    "EncDec": EncDec,
    "UNet": UNet,
}
model = models["EncDec"]().to(device)

summary(model, (3, 256, 256))
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), learning_rate)

loss_fns = {
    "BCELoss": BCELoss,
    "DiceLoss": DiceLoss,
    "FocalLoss": FocalLoss,
    "BCELoss_TotalVariation": BCELoss_TotalVariation,
}
loss_fn = loss_fns["BCELoss"]()

epochs = 20

# Training loop
for _epoch in tqdm(range(epochs)):
    tic = time()

    # Training phase
    model.train()
    train_loss = 0
    for X_batch, y_true in train_loader:
        X_batch_gpu = X_batch.to(device)
        y_true_gpu = y_true.to(device)

        # set parameter gradients to zero
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = model(X_batch_gpu)
        # IMPORTANT NOTE: Check whether y_pred is normalized or unnormalized
        # and whether it makes sense to apply sigmoid or softmax.
        loss = loss_fn(y_pred, y_true_gpu)  # forward-pass
        loss.backward()  # backward-pass
        optimizer.step()  # update weights

        # calculate metrics to show the user
        train_loss += loss.item() / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_true in val_loader:
            X_batch_gpu = X_batch.to(device)
            y_true_gpu = y_true.to(device)

            y_pred = model(X_batch_gpu)
            loss = loss_fn(y_pred, y_true_gpu)
            val_loss += loss.item() / len(val_loader)

    toc = time()

# Save the model
torch.save(model.state_dict(), "Project_3/model/checkpoints/encdec.pth")

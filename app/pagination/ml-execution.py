import streamlit as st


# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, Concatenate, Dropout, Layer
# from tensorflow.keras.layers import ReLU, LeakyReLU


import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
DATA_PATH = "../data/"

import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


with h5py.File(DATA_PATH + 'bkg_dataset.h5', 'r') as file:
    X_train = np.array(file['X_train'])
    X_test = np.array(file['X_test'])
    
with h5py.File(DATA_PATH + 'signal_dataset.h5', 'r') as file:
    signal_test_data = np.array(file['Data'])

scaler = StandardScaler()
# _ = scaler.fit(x_bkg)
# x_bkg_scaled = scaler.transform(x_bkg)
# x_sig_scaled = scaler.transform(x_sig)
    
        
# define training, test and validation datasets
#X_train, X_test = train_test_split(x_bkg_scaled, test_size=0.2, shuffle=True)

# print("Training data shape = ",X_train.shape)    
# with h5py.File('bkg_dataset.h5', 'w') as h5f:
#     h5f.create_dataset('X_train', data = X_train)
#     h5f.create_dataset('X_test', data = X_test)
    
# with h5py.File('signal_dataset.h5', 'w') as h5f2:
#     h5f2.create_dataset('Data', data = x_sig_scaled)        


import torch
from torch import nn

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms


import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 32),                     # Equivalent to Dense(32)
            nn.BatchNorm1d(32),                             # BatchNormalization
            nn.LeakyReLU(0.3),                              # LeakyReLU activation

            nn.Linear(32, 16),                              # Dense(16)
            nn.BatchNorm1d(16),                             # BatchNormalization
            nn.LeakyReLU(0.3),                              # LeakyReLU activation

            nn.Linear(16, latent_dim)                       # Compress to latent space
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),                      # Start decoding from latent space
            nn.BatchNorm1d(16),                             # BatchNormalization
            nn.LeakyReLU(0.3),                              # LeakyReLU activation

            nn.Linear(16, 32),                              # Dense(32)
            nn.BatchNorm1d(32),                             # BatchNormalization
            nn.LeakyReLU(0.3),                              # LeakyReLU activation

            nn.Linear(32, input_shape)                      # Reconstruct input shape
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# Summary (Optional, requires the torchsummary package)
#from torchsummary import summary
#summary(autoencoder, input_size=(input_shape,))


# Model parameters
input_shape = X_train.shape[1]
latent_dim = 3

# Create the autoencoder model
autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent_dim)
print(autoencoder)



import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Check if CUDA is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
autoencoder = Autoencoder(input_shape=input_shape, latent_dim=latent_dim).to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-5)

# Callbacks equivalent
class EarlyStopping:
    def __init__(self, patience=10, delta=0, restore_best_weights=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model_weights)
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()

# ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, min_lr=1e-6)

# Training parameters
EPOCHS = 5
BATCH_SIZE = 1024
early_stopping = EarlyStopping(patience=10, delta=1e-4, restore_best_weights=True)

if st.button("Train"):
    train = True

# Early Stopping Callback

# # Training loop
# if train:
    train_data = torch.tensor(X_train, dtype=torch.float32).to(device)
    val_split = int(0.8 * len(train_data))
    train_set, val_set = torch.utils.data.random_split(train_data, [val_split, len(train_data) - val_split])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        autoencoder.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = autoencoder(inputs)

            # Calculate loss
            loss = criterion(outputs, inputs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation loop
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.to(device)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        # ReduceLROnPlateau and EarlyStopping callbacks
        scheduler.step(val_loss / len(val_loader))
        early_stopping(val_loss / len(val_loader), autoencoder)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Save the model
    torch.save(autoencoder.state_dict(), 'baseline_ae.pth')
else:
    # Load the model
    autoencoder.load_state_dict(torch.load(DATA_PATH + 'baseline_ae.pth'))



if st.button('Run inference'):
    
    # Inference
    autoencoder.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    signal_test_tensor = torch.tensor(signal_test_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        bkg_prediction = autoencoder(X_test_tensor).cpu().numpy()
        signal_prediction = autoencoder(signal_test_tensor).cpu().numpy()

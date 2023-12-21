import h5py
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
import os
from sklearn.manifold import  TSNE
import matplotlib.pyplot as plt


file_path = 'magfield_96.h5'

wandb.init(project='VAE parts',entity='zhulin010216')
with h5py.File(file_path, 'r') as hdf:
    dataset_name = 'field'
    data = hdf[dataset_name][:]


print(data.shape)
if not os.path.exists('latent_space_visualizations'):
    os.makedirs('latent_space_visualizations')

def visualize_latent_space(encoder, data_loader, epoch, device):
    encoder.eval()
    latent_space_points = []
    with torch.no_grad():
        for batch_idx, (data,) in enumerate(data_loader):
            data = data.to(device)
            mu, logvar = encoder(data)
            z = encoder.reparameterize(mu, logvar)
            latent_space_points.append(z.cpu().numpy())

    # Concatenate all collected points into a single NumPy array
    latent_space_points = np.concatenate(latent_space_points, axis=0)

    # Use t-SNE to project the points down to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(latent_space_points)

    # Now, we can visualize the points with matplotlib
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title(f't-SNE projection of the latent space - Epoch {epoch}')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')

    # Save the plot
    plt.savefig(f'latent_space_visualizations/tsne_epoch_{epoch}.png')
    plt.close()

# Define the ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)



class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, cnum, latent_dim, num_residual_layers):
        super(VAE_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim[0], cnum, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum, cnum * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum * 2, cnum * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum * 4, cnum * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cnum * 8, cnum * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25)
        )

        conv_output_size = cnum * 16 * 3 * 3  # Adjusted for the new input size

        self.fc_mean = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        #x = self.residual_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class VAE_Decoder(nn.Module):
    def __init__(self, input_dim, cnum, latent_dim, num_residual_layers):
        super(VAE_Decoder, self).__init__()

        H_end = 3  # Adjusted for the new input size
        W_end = 3  # Adjusted for the new input size
        self.feature_map_height = H_end
        self.feature_map_width = W_end

        fc_output_size = cnum * 16 * H_end * W_end

        self.fc = nn.Linear(latent_dim, fc_output_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(cnum * 16, cnum * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnum * 8, cnum * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnum * 4, cnum * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnum * 2, cnum, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(cnum),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(cnum, input_dim[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, cnum * 16, self.feature_map_height, self.feature_map_width)
        #z = self.residual_layers(z)
        reconstruction = self.decoder(z)
        return reconstruction

# MSE
def mse_loss(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='mean')

# KL
def kl_divergence_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()

def calculate_relative_error(pred, target):
    relative_error = torch.norm(pred - target, dim=1) / (torch.norm(target, dim=1) + 1e-6)
    return torch.median(relative_error) * 100


input_dim = (3, 96, 96)
cnum = 64
latent_dim = 128
num_residual_layers = 2

data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
train_data, test_data = train_test_split(data, test_size=1/3, random_state=42)


epochs = 400
learning_rate = 1e-4
batch_size = 60
#record the parameter through the wandb
wandb.config.batch_size = batch_size
wandb.config.learning_rate = learning_rate

train_tensor = torch.from_numpy(train_data).float()
test_tensor = torch.from_numpy(test_data).float()


train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


encoder = VAE_Encoder(input_dim, cnum, latent_dim, num_residual_layers)
decoder = VAE_Decoder(input_dim, cnum, latent_dim, num_residual_layers)


params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = Adam(params, lr=learning_rate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

for epoch in range(epochs):
    encoder.train()
    decoder.train()

    train_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    rmse_loss = 0.0
    mae_loss = 0.0
    test_rmse_loss = 0.0
    test_mae_loss = 0.0

    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()


        mu, logvar = encoder(data)
        z = encoder.reparameterize(mu, logvar)


        recon_data = decoder(z)


        recon = mse_loss(recon_data, data)
        kl_div = kl_divergence_loss(mu, logvar)


        loss = recon + 0.0005*kl_div
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl_div.item()

        rmse = torch.sqrt(torch.mean((recon_data - data) ** 2))
        mae = torch.mean(torch.abs(recon_data - data))

        rmse_loss += rmse.item()
        mae_loss += mae.item()


    encoder.eval()
    decoder.eval()
    errors = []
    with torch.no_grad():
        for batch in test_loader:
            input_data = batch[0].to(device)
            mu, logvar = encoder(input_data)
            z = encoder.reparameterize(mu, logvar)
            recon_data = decoder(z)
            error = calculate_relative_error(recon_data, input_data)
            errors.append(error.item())

            test_rmse = torch.sqrt(torch.mean((recon_data - input_data) ** 2))
            test_mae = torch.mean(torch.abs(recon_data - input_data))

            test_rmse_loss += test_rmse.item()
            test_mae_loss += test_mae.item()

    avg_test_rmse = test_rmse_loss / len(test_loader)
    avg_test_mae = test_mae_loss / len(test_loader)
    median_error = np.median(errors)
    avg_rmse = rmse_loss / len(train_loader)
    avg_mae = mae_loss / len(train_loader)
    wandb.log(
        {"Train Loss": train_loss, "Reconstruction Loss": recon_loss, "KL Divergence Loss": kl_loss, "RMSE": avg_rmse,
         "MAE": avg_mae})

    if epoch % 10 == 0:
        visualize_latent_space(encoder, test_loader, epoch, device)



    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}, '
          f'Reconstruction Loss: {recon_loss / len(train_loader.dataset)}, '
          f'KL Divergence Loss: {kl_loss / len(train_loader.dataset)}, '
          f'Median Relative Error on Test Set: {median_error}%'
          f'Average RMSE: {avg_rmse}, Average MAE: {avg_mae}'
          f'Average RMSE on Test Set: {avg_test_rmse}, Average MAE on Test Set: {avg_test_mae}')

torch.save(encoder.state_dict(), "VAE-Encoder.pth")
wandb.save("VAE-Encoder.pth")
torch.save(decoder.state_dict(), "VAE-Decoder.pth")
wandb.save("VAE-Decoder.pth")
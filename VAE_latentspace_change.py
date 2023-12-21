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
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from sklearn.manifold import TSNE

file_path = 'magfield_96.h5'

wandb.init(project='VAE parts',entity='zhulin010216')
with h5py.File(file_path, 'r') as hdf:
    dataset_name = 'field'
    data = hdf[dataset_name][:]


print(data[0].shape)
print(data.shape)
#print(hdf[dataset_name][1].shape)  error

if not os.path.exists('latentspace_change_projected_to_2D'):
    os.makedirs('latentspace_change_projected_to_2D')


#def visualize_latent_space_1d(encoder, data_loader, epoch, device, dim=0):
#    encoder.eval()
#    latent_space_points = []
#    with torch.no_grad():
#        for batch_idx, (data,) in enumerate(data_loader):
#            data = data.to(device)
#            mu, logvar = encoder(data)
#            z = encoder.reparameterize(mu, logvar)
#            latent_space_points.append(z.cpu().numpy()[:, dim])

    # Concatenate all collected points into a single NumPy array
#    latent_space_points = np.concatenate(latent_space_points, axis=0)

#    plt.figure(figsize=(10, 6))
#    sns.kdeplot(latent_space_points, fill=True)  # 使用KDE进行密度估计
#    plt.title(f'Latent Space Distribution along Dimension {dim} - Epoch {epoch}')
#    plt.xlabel(f'Dimension {dim} Value')
#    plt.ylabel('Density')

    # Save the plot
 #   plt.savefig(f'latentspace_change_projected_to_2D/density_epoch_{epoch}_dim_{dim}.png')
 #   plt.close()

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


def plot_magfield(
        field: np.ndarray, vmax: float = 1
) -> None:

    plt.clf()
    labels = ['Bx-field', 'By-field', 'Bz-field']
    nrows = 3 if len(field.shape) == 4 else 1  # set the number of rows in the subplot grid
    fig, axes = plt.subplots(nrows=nrows, ncols=3, sharex=True,  # create a grid of subplots with'nrows'and'ncols'
                             sharey=True, figsize=(15, 10))
    norm = colors.Normalize(vmin=-vmax, vmax=vmax)  # create a color normalization object that maps the data values to the color map

    if len(field.shape) == 3:  # it's a 2D vector
        for i, comp in enumerate(field):
            ax = axes.flat[i]
            im = ax.imshow(comp, cmap='bwr', norm=norm, origin="lower")
            ax.set_title(labels[i])

    elif len(field.shape) == 4:  # it's a 3D vector
        for i, z in enumerate([0, 1, 2]):
            #for j, comp in enumerate(field[z, :, :, :]):
            for j, comp in enumerate(field[:, :, :, z]):
                ax = axes.flat[i * 3 + j]
                im = ax.imshow(comp, cmap='bwr', norm=norm, origin="lower")
                ax.set_title(labels[j] + f'@{z + 1}')

    else:
        raise NotImplementedError()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.345, 0.015, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


#def plot_magfield_interpolations(encoder, decoder, data_loader, device, num_interpolations=20, vmax=0.25):
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
    plt.savefig(f'latentspace_change_projected_to_2D/tsne_epoch_{epoch}.png')
    plt.close()


######START#######
input_dim = (3, 96, 96)
cnum = 64
latent_dim = 128
num_residual_layers = 2

data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
train_data, test_data = train_test_split(data, test_size=1/3, random_state=42)

epochs = 200
learning_rate = 1e-4
batch_size = 60
#record the parameter through the wandb
wandb.config.batch_size = batch_size
wandb.config.learning_rate = learning_rate

train_tensor = torch.from_numpy(train_data).float()
test_tensor = torch.from_numpy(test_data).float()

train_dataset = TensorDataset(train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)   #train_loader：   DataLoader按照 batch_size把train_dataset分为不同批次，然后放到train_loader中
test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(test_tensor.shape)

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
    print(epoch)
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

        #if epoch % 20 == 0:
        #    for i in range(2):
                # 选择数据点

                # 通过VAE转换数据点
        #        encoded_points = encoder(selected_points)
        #        decoded_points = decoder(encoded_points)

        #        plot_magfield(decoded_points,vmax=0.25)
        #        plt.title('decoder Data Points')
        #        plt.savefig(f'magfield_visualizations_graph/decoder data {i}.png')

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

#            if epoch%50 == 0:
#                selected_points = input_data[0].cpu().numpy()  # (3,96,96)
#                plot_magfield(selected_points, vmax=0.25)
#                plt.title('Original Data Points')
#                plt.savefig(f'latentspace_change_smooth/original data{epoch}.png')

#                decoded_points = recon_data[0].cpu().numpy()
#                plot_magfield(decoded_points, vmax=0.25)
#                plt.title('decoder Data Points')
#                plt.savefig(f'latentspace_change_smooth/decoder data {epoch}.png')
#                plt.close('all')

            while(epoch==10):
                # plot_magfield_interpolations(encoder, decoder, test_loader, device)
                #print("interpolation_begin")
                num_interpolations = 10
                vmax = 0.25
                data_loader = test_loader
                v1, v2 = next(iter(data_loader))[0][:2].to(device)
                v1 = v1.unsqueeze(0)  # 添加批次维度
                v2 = v2.unsqueeze(0)  # 添加批次维度    (1,3,96,96)
                #print(v1.shape)
                #print("debug0")
                latent_space_points = []
                with torch.no_grad():
                    mu_v1, logvar_v1 = encoder(v1)
                    mu_v2, logvar_v2 = encoder(v2)
                    z_v1 = encoder.reparameterize(mu_v1, logvar_v1)
                    z_v2 = encoder.reparameterize(mu_v2, logvar_v2)

                t_values = np.linspace(0, 1, num_interpolations)
                interpolated_points = []
                #plt.figure(figsize=(20, 10))
                #print("debug1")
                for i, t in enumerate(t_values, 1):
                    with torch.no_grad():
                        v_prime = (1 - t) * z_v1 + t * z_v2     #shape（1，3，96，96）
                        #print("debug2")
                        reconstructed = decoder(v_prime)[0].cpu().numpy()
                        # Concatenate all collected points into a single NumPy array

                        interpolated_points.append(v_prime.cpu().numpy())

                interpolated_points = np.concatenate(interpolated_points, axis=0)

                # Use t-SNE to project the points down to 2 dimensions
                tsne = TSNE(n_components=2, random_state=42)
                tsne_results = tsne.fit_transform(interpolated_points)

                # Now, we can visualize the points with matplotlib
                plt.figure(figsize=(10, 8))
                plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
                plt.title(f't-SNE projection of the latent space - Epoch {epoch}-λ {i}')
                plt.xlabel('t-SNE feature 1')
                plt.ylabel('t-SNE feature 2')
                plt.savefig(f'latentspace_change_projected_to_2D/latent space smooth change_{i}.png')
                #print("debug3")
                #plt.subplot(2, int(np.ceil(num_interpolations / 2)), i)

                #plot_magfield(reconstructed, vmax=vmax)

                #plt.title(f't = {t:.2f}')
                #plt.axis('off')
                #plt.savefig(f'latentspace_change_smooth/latent space smooth change_{i}.png')
                #plt.show()
                #plt.close()



    avg_test_rmse = test_rmse_loss / len(test_loader)
    avg_test_mae = test_mae_loss / len(test_loader)
    median_error = np.median(errors)
    avg_rmse = rmse_loss / len(train_loader)
    avg_mae = mae_loss / len(train_loader)

    #wandb
    #if epoch%5==0:
    #    wandb.log(
    #    {"Train Loss": train_loss, "Reconstruction Loss": recon_loss, "KL Divergence Loss": kl_loss, "RMSE": avg_rmse,
    #     "MAE": avg_mae})

#    if epoch % 10 == 0:
#        visualize_latent_space_1d(encoder, test_loader, epoch, device, dim = 0)



    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}, '
          f'Reconstruction Loss: {recon_loss / len(train_loader.dataset)}, '
          f'KL Divergence Loss: {kl_loss / len(train_loader.dataset)}, '
          f'Median Relative Error on Test Set: {median_error}%'
          f'Average RMSE: {avg_rmse}, Average MAE: {avg_mae}'
          f'Average RMSE on Test Set: {avg_test_rmse}, Average MAE on Test Set: {avg_test_mae}')

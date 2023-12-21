import h5py
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
import numpy as np



file_path = '/zhome/f6/7/202280/dataset/magfield_96.h5'
with h5py.File(file_path, 'r') as hdf:
    dataset_name = 'field'
    data = hdf[dataset_name][:]

print(data.shape)


class Generator(nn.Module):
    def __init__(self, z_dimension, input_dim, cnum):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dimension, cnum * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(cnum * 16),
            nn.ReLU(True),
            # 尺寸: cnum*16 x 4 x 4
            nn.ConvTranspose2d(cnum * 16, cnum * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 8),
            nn.ReLU(True),
            # 尺寸: cnum*8 x 8 x 8
            nn.ConvTranspose2d(cnum * 8, cnum * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4),
            nn.ReLU(True),
            # 尺寸: cnum*4 x 16 x 16
            nn.ConvTranspose2d(cnum * 4, cnum * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 2),
            nn.ReLU(True),
            # 尺寸: cnum*2 x 32 x 32
            nn.ConvTranspose2d(cnum * 2, input_dim, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
            # 最终尺寸: input_dim x 96 x 96
        )

    def forward(self, x):
        x = self.gen(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, input_dim, cnum):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(input_dim, cnum, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸: cnum x 48 x 48
            nn.Conv2d(cnum, cnum * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸: cnum*2 x 24 x 24
            nn.Conv2d(cnum * 2, cnum * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸: cnum*4 x 12 x 12
            nn.Conv2d(cnum * 4, cnum * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cnum * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 尺寸: cnum*8 x 6 x 6
            nn.Conv2d(cnum * 8, 1, kernel_size=5, stride=2, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.dis(x)
        return x.view(-1, 1)

    # Training


input_dim = 3
cnum = 64
z_dimension = 100
num_epoch = 150
batch_size = 50

from torch.utils.data import DataLoader, TensorDataset

if not isinstance(data, torch.Tensor):
    data = torch.tensor(data, dtype=torch.float32)

dataset = TensorDataset(data)

# Create DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
generator = Generator(z_dimension, input_dim, cnum).to(device)
discriminator = Discriminator(input_dim, cnum).to(device)


# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0003, betas=(0, 0.9))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0, 0.9))

# For calculating the gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).expand_as(real_samples).to(real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(real_samples.device)
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def curl(field):
    Fx_y = np.gradient(field[0], axis=0)
    Fy_x = np.gradient(field[1], axis=1)
    curl_vec = Fy_x - Fx_y
    return curl_vec

def div(field):
    Fx_x = np.gradient(field[0], axis=1)
    Fy_y = np.gradient(field[1], axis=0)
    div = np.stack([Fx_x, Fy_y], axis=0)
    return div.sum(axis=0)

# Training loop
lambda_gp = 10
n_critic = 3 # The generator is updated every n_critic steps
for epoch in range(num_epoch):
    for i, batch in enumerate(data_loader):  # Replace with your data_loader

        imgs = batch[0]
        # Transfer images to the appropriate device
        real_imgs = imgs.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if i % n_critic == 0:
            d_optimizer.zero_grad()

            # Sample noise as generator input
            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, z_dimension)))).to(device)
            z = z.view(z.size(0), z.size(1), 1, 1)

            # Generate a batch of images
            fake_imgs = generator(z).detach()
            # Real images
            real_validity = discriminator(real_imgs)
            # Fake images
            fake_validity = discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            # Adversarial loss
            d_loss = torch.mean(real_validity-fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(z)

        gen_imgs_2d = gen_imgs[:, :2, :, :].detach().cpu().numpy()
        # Calculate physical correctness
        curl_value = curl(gen_imgs_2d)
        div_value = div(gen_imgs_2d)
        curl_value_tensor = torch.tensor(curl_value, device=device)
        div_value_tensor = torch.tensor(div_value, device=device)

        # Loss measures generator's ability to fool the discriminator
        g_loss = -torch.mean(discriminator(gen_imgs))+(curl_value_tensor).mean()+(div_value_tensor).mean()

        g_loss.backward()
        g_optimizer.step()

        # Print out the losses, discriminator scores, and save images periodically
        print(f"[Epoch {epoch}/{num_epoch}] [Batch {i}/{len(data_loader)}] "
              f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}] "
              f"[Real Score: {real_validity.mean().item()}] [Fake Score: {fake_validity.mean().item()}]"
              f"Curl: {np.mean(curl_value)} Div: {np.mean(div_value)}")

        batches_done = epoch * len(data_loader) + i




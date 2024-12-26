# coding a GAN
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
device = "cuda" if torch.cuda.is_available else "cpu"
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        return self.disc(x)
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.gen(x)
lr = 3e-4
z_dim = 64
img_dim = 784 # 28*28*1
batch_size = 32
num_epochs = 200
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn(batch_size, z_dim).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))]
)
dataset = datasets.MNIST(root = "data",transform = transforms, download = True )
loader = DataLoader(dataset, batch_size= batch_size, shuffle = True)
disc_optim = optim.Adam(disc.parameters(), lr= lr)
gen_optim = optim.Adam(gen.parameters(), lr= lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step =0
for epoch in range(num_epochs):
    for batch, (X, y) in enumerate(loader):
        X= X.view(-1, 784).to(device) # rehshape
        batch_size = X.shape[0]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(X).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real+loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        disc_optim.step()
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()
        if batch %1875 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch}/{len(loader)} \
                      Loss D: {loss_disc:.8f}, loss G: {loss_gen:.8f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = X.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1

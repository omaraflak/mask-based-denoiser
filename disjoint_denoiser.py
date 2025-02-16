import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Focuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = self.generate_masks().to(device)
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.masks.shape[0]),
            nn.Sigmoid()
        )
        self.weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weights = self.model(x)
        # weights: (BATCH, masks_count)
        # masks: (masks_count, X_SIZE)
        # result: (BATCH, X_SIZE)
        return torch.einsum('ij,jk->ijk', self.weights, self.masks).sum(dim=1)

    def generate_masks(self) -> torch.Tensor:
        masks = torch.zeros(49, 28, 28)
        idx = 0
        for i in range(7):
            for j in range(7):
                masks[idx, i*4:(i+1)*4, j*4:(j+1)*4] = 1
                idx += 1
        return masks.view(49, -1)


def add_gaussian_noise(images: torch.Tensor, noise_factor: float) -> torch.Tensor:
    noisy_images = images + noise_factor * torch.randn(*images.shape)
    return torch.clamp(noisy_images, 0., 1.)


def train_denoiser(epochs: int, batch_size: int, learning_rate: float, noise_factor: float) -> Denoiser:
    # dataset
    mnist_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True)

    # model, loss and optimizer
    model = Denoiser().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for images, _ in train_loader:
            images = images.view(-1, 784)
            noisy_images = add_gaussian_noise(images, noise_factor)

            images = images.to(device)
            noisy_images = noisy_images.to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return model


def train_focuser(
    denoiser: Denoiser,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    noise_factor: float
) -> Focuser:
    # dataset
    mnist_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True)

    # model, loss and optimizer
    model = Focuser().to(device)
    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # don't train the original model
    model.train(True)
    denoiser.train(False)

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in train_loader:
            images = images.view(-1, 784)
            noisy_images = add_gaussian_noise(images, noise_factor).to(device)

            merged_masks = model(noisy_images)
            masked_noisy_images = merged_masks * noisy_images
            unmasked_and_denoised_images = denoiser(masked_noisy_images)

            # minimize the denoiser error for masked inputs
            loss = mse(unmasked_and_denoised_images, images.to(device))

            # minimize number of parts of the image to keep, i.e. minimize mask weights
            loss += temperature * \
                mse(model.weights, torch.zeros_like(model.weights))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return model


def plot_evaluations(
    denoiser: Denoiser,
    focuser: Focuser,
    samples: int,
    noise_factor: float,
    filename: str
):
    test_mnist_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(
        test_mnist_dataset, batch_size=samples, shuffle=True)

    focuser.train(False)
    denoiser.train(False)

    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.view(-1, 784)

        # inputs
        noisy = add_gaussian_noise(images, noise_factor).to(device)

        # predict images
        merged_masks = focuser(noisy)
        denoised = denoiser(noisy).cpu()
        masked_pairs = (merged_masks * noisy).cpu()
        unmasked_and_denoised = denoiser(merged_masks * noisy).cpu()
        merged_masks = merged_masks.cpu()
        noisy = noisy.cpu()

        # Plot results
        plt.figure(figsize=(15, 10))
        for i in range(samples):
            # Noisy
            plt.subplot(5, samples, i + 1)
            plt.imshow(noisy[i].view(28, 28), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('x')

            # Denoised
            plt.subplot(5, samples, i + 1 + samples)
            plt.imshow(denoised[i].view(28, 28), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('f(x)')

            # Mask
            plt.subplot(5, samples, i + 1 + 2 * samples)
            plt.imshow(merged_masks[i].view(28, 28),
                       vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('p(x)')

            # Masked pairs
            plt.subplot(5, samples, i + 1 + 3 * samples)
            plt.imshow(masked_pairs[i].view(28, 28),
                       vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('m(x)')

            # Unmasked + Denoised
            plt.subplot(5, samples, i + 1 + 4 * samples)
            plt.imshow(unmasked_and_denoised[i].view(
                28, 28), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('f(m(x))')

        plt.tight_layout()
        plt.title('white: keep, black: remove')
        plt.savefig(filename)


def plot_masks(focuser: Focuser, samples: int, filename: str):
    plt.figure(figsize=(25, 10))

    for i in range(samples):
        img = focuser.masks[i].view(28, 28).cpu()
        plt.subplot(1, samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)


def measure_models_difference(denoiser: Denoiser, focuser: Focuser, noise_factor: float) -> float:
    test_mnist_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_mnist_dataset, batch_size=256, shuffle=True)
    focuser.train(False)
    denoiser.train(False)

    difference = 0

    for images, _ in test_loader:
        images = images.view(-1, 784)
        noisy_images = add_gaussian_noise(images, noise_factor).to(device)
        merged_masks = focuser(noisy_images)

        denoised_images = denoiser(noisy_images)
        denoised_masked_images = denoiser(noisy_images * merged_masks)

        difference += F.mse_loss(denoised_images, denoised_masked_images)

    difference /= len(test_loader)
    return difference


def main():
    denoiser = train_denoiser(
        epochs=10,
        batch_size=256,
        learning_rate=0.001,
        noise_factor=0.3
    )
    focuser = train_focuser(
        denoiser,
        epochs=10,
        batch_size=256,
        learning_rate=0.001,
        temperature=0.1,
        noise_factor=0.3
    )
    plot_evaluations(
        denoiser,
        focuser,
        samples=10,
        noise_factor=0.3,
        filename='evaluations.png'
    )
    plot_masks(
        focuser,
        samples=10,
        filename='masks.png'
    )
    diff = measure_models_difference(denoiser, focuser, noise_factor=0.3)
    print(f"MSE(f(x), f(m(x))): {diff}")


if __name__ == '__main__':
    main()

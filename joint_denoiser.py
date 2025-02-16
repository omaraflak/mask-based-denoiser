import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Denoiser(nn.Module):

    def __init__(self):
        super().__init__()

        # (x, y) 1568 = 28 * 28 * 2
        self.encoder = nn.Sequential(
            nn.Linear(1568, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1568),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Focuser(nn.Module):

    def __init__(self):
        super().__init__()
        self.masks = self.generate_masks().to(device)
        self.model = nn.Sequential(
            nn.Linear(1568, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.masks.shape[0]),
            nn.Sigmoid(),
        )
        self.weights = None

    def forward(self, x):
        self.weights = self.model(x)
        # weights: (BATCH, masks_count)
        # masks: (masks_count, XY_SIZE)
        # result: (BATCH, XY_SIZE)
        return torch.einsum('ij,jk->ijk', self.weights, self.masks).sum(dim=1)

    def generate_masks(self) -> torch.Tensor:
        # we flatten by rows, so (x, y) have to be stacked vertically
        masks = torch.zeros(98, 56, 28)
        idx = 0
        for i in range(14):
            for j in range(7):
                masks[idx, i * 4: (i + 1) * 4, j * 4: (j + 1) * 4] = 1
                idx += 1
        return masks.view(98, -1)


def add_gaussian_noise(data: torch.Tensor, noise_factor: float) -> torch.Tensor:
    noisy = data + noise_factor * torch.randn(*data.shape)
    return torch.clamp(noisy, 0, 1)


def train_models(
    epochs: int,
    batch_size: int,
    learning_rate: int,
    true_task_importance: float,
    temperature: float,
    noise_factor: float
) -> tuple[Denoiser, Focuser, dict[str, list[float]]]:
    # dataset
    mnist_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=True)

    # model, loss and optimizer
    denoiser = Denoiser().to(device)
    focuser = Focuser().to(device)
    mse = nn.MSELoss()

    params = itertools.chain(denoiser.parameters(), focuser.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    losses = {'total': [], 'denoise': [], 'unmask': [], 'regularizer': []}
    for epoch in range(epochs):
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0

        for images, _ in train_loader:
            images = images.view(-1, 784)
            noisy_images = add_gaussian_noise(images, noise_factor)

            # build (x, y) pairs
            xy = torch.concat((noisy_images, images), dim=1).to(device)

            # f((x, 0))
            empty_mask = torch.zeros_like(noisy_images)
            masked_output_pairs = torch.concat(
                (noisy_images, empty_mask), dim=1)
            denoised_pairs = denoiser(masked_output_pairs.to(device))
            loss1 = mse(denoised_pairs, xy)

            # f(m(x, y))
            merged_masks = focuser(xy)
            masked_pairs = merged_masks * xy
            unmasked_and_denoised_pairs = denoiser(masked_pairs)
            loss2 = mse(unmasked_and_denoised_pairs, xy)

            # minimize number of masks applied (a mask represents a part of the input to keep)
            loss3 = mse(focuser.weights, torch.zeros_like(focuser.weights))

            # total loss
            loss = true_task_importance * loss1 + loss2 + temperature * loss3

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            total_loss3 += loss3.item()
            losses['total'].append(loss.item())
            losses['denoise'].append(loss1.item())
            losses['unmask'].append(loss2.item())
            losses['regularizer'].append(loss3.item())

        avg_loss = total_loss / len(train_loader)
        avg_loss1 = total_loss1 / len(train_loader)
        avg_loss2 = total_loss2 / len(train_loader)
        avg_loss3 = total_loss3 / len(train_loader)
        print(
            f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Task Loss:'
            f' {avg_loss1:.4f}, Unmask Loss: {avg_loss2:.4f}, Mask: {avg_loss3:.4f}'
        )

    return denoiser, focuser, losses


def plot_losses(losses: dict[str, list[float]], filename: str):
    plt.figure(figsize=(15, 10))
    plt.plot(losses['total'], label='Total Loss')
    plt.plot(losses['denoise'], label='Denoise Loss')
    plt.plot(losses['unmask'], label='Unmask Loss')
    plt.plot(losses['regularizer'], label='Regularizer Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)


def plot_evaluations(
    denoiser: Denoiser,
    focuser: Focuser,
    noise_factor: float,
    samples: int,
    filename: str
):
    # load dataset
    test_mnist_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        test_mnist_dataset, batch_size=samples, shuffle=True)

    denoiser.train(False)
    focuser.train(False)

    print('masking:')
    print('black -> hide')
    print('white -> keep')

    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.view(-1, 784)

        noisy_images = add_gaussian_noise(images, noise_factor)
        empty_mask = torch.zeros_like(noisy_images)
        x0 = torch.concat((noisy_images, empty_mask), dim=1).to(device)

        merged_masks = focuser(x0)
        denoised = denoiser(x0).cpu()
        unmasked_and_denoised = denoiser(merged_masks * x0).cpu()
        masked_pairs = (merged_masks * x0).cpu()
        merged_masks = merged_masks.cpu()

        # Plot results
        plt.figure(figsize=(15, 10))
        for i in range(samples):
            # Noisy
            plt.subplot(5, samples, i + 1)
            plt.imshow(noisy_images[i].view(28, 28),
                       vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('x')

            # Denoised
            plt.subplot(5, samples, i + 1 + samples)
            plt.imshow(denoised[i].view(56, 28), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('f((x, 0))')

            # Mask
            plt.subplot(5, samples, i + 1 + 2 * samples)
            plt.imshow(merged_masks[i].view(56, 28),
                       vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('p(x,0)')

            # Masked pairs
            plt.subplot(5, samples, i + 1 + 3 * samples)
            plt.imshow(masked_pairs[i].view(56, 28),
                       vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('m(x, 0)')

            # Unmasked + Denoised
            plt.subplot(5, samples, i + 1 + 4 * samples)
            plt.imshow(unmasked_and_denoised[i].view(
                56, 28), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('f(m(x, 0))')

        plt.tight_layout()
        plt.title("white: keep, black: remove")
        plt.savefig(filename)


def plot_masks(focuser: Focuser, samples: int, filename: str):
    plt.figure(figsize=(25, 10))

    for i in range(samples):
        img = focuser.masks[i].view(56, 28).cpu()
        plt.subplot(1, samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)


def main():
    denoiser, focuser, losses = train_models(
        epochs=10,
        batch_size=256,
        learning_rate=0.001,
        true_task_importance=1.3,
        temperature=0.1,
        noise_factor=0.3
    )
    plot_losses(losses, filename='losses.png')
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


if __name__ == "__main__":
    main()

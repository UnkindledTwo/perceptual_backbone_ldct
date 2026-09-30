"""Train UNet model for image denoising."""

import numpy as np
import torch
import torch.nn as nn
from config import CONFIG
from model import UNet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DenoisingDataset(Dataset):
    def __init__(self, data_dir, noise_key: str):
        self.files = sorted(data_dir.glob("*.npz"))
        self.noise_key = noise_key

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        noisy = data[self.noise_key].astype(np.float32)
        gt = data["gt_clipped"].astype(np.float32)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)
        return noisy, gt


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for noisy, gt in pbar:
        noisy, gt = noisy.to(device), gt.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    psnr_values, ssim_values = [], []
    with torch.no_grad():
        for noisy, gt in tqdm(loader, desc="Val", leave=False):
            noisy, gt = noisy.to(device), gt.to(device)
            output = model(noisy)
            loss = criterion(output, gt)
            total_loss += loss.item()
            pred = output.clamp(0, 1).cpu().numpy()
            target = gt.cpu().numpy()
            for i in range(pred.shape[0]):
                p = pred[i, 0]
                t = target[i, 0]
                psnr_values.append(peak_signal_noise_ratio(t, p, data_range=1.0))
                ssim_values.append(structural_similarity(t, p, data_range=1.0))
    avg_loss = total_loss / len(loader)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    return avg_loss, avg_psnr, avg_ssim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = DenoisingDataset(CONFIG.train_dir, CONFIG.noise_key)
    val_dataset = DenoisingDataset(CONFIG.val_dir, CONFIG.noise_key)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size, shuffle=False, num_workers=4)

    model = UNet(CONFIG.in_channels, CONFIG.out_channels, CONFIG.features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.num_epochs)

    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(CONFIG.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{CONFIG.num_epochs} | Train Loss: {train_loss:.6f}"
            f" | Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG.checkpoint_dir / "best_model.pth")
            print(f"  Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

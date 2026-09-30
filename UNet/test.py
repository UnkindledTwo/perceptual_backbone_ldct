"""Run trained UNet on test set."""

import numpy as np
import torch
from config import CONFIG
from model import UNet
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(CONFIG.in_channels, CONFIG.out_channels, CONFIG.features).to(device)
    checkpoint_path = CONFIG.checkpoint_dir / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    input_files = sorted(CONFIG.test_dir.glob("*.npz"))

    with torch.no_grad():
        for npz_path in tqdm(input_files, desc="UNet"):
            data = np.load(npz_path)
            noisy = data[CONFIG.noise_key].astype(np.float32)
            noisy_tensor = torch.from_numpy(noisy).unsqueeze(0).unsqueeze(0).to(device)
            output = model(noisy_tensor)
            denoised = output.squeeze().cpu().numpy().astype(np.float32)
            denoised = np.clip(denoised, 0, 1)
            np.savez_compressed(CONFIG.output_dir / npz_path.name, denoised=denoised)


if __name__ == "__main__":
    main()

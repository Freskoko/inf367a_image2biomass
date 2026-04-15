import torch
import numpy as np
from torchvision.utils import save_image
from src.main.ccgan_improved.models import cont_cond_cnn_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the pre-trained weights
# Replace with your actual .pth path
# checkpoint_path = "../output/saved_models/ckpt_CcGAN_niters_400_seed_2020_hard_0.06381135070109936_0.41081081081081083.pth"
checkpoint_path = "Cell-200/Cell-200_64x64/CcGAN-improved/output/saved_models/ckpt_CcGAN_niters_400_seed_2020_hard_0.06381135070109936_0.41081081081081083.pth"
checkpoint = torch.load(checkpoint_path)

# 2. Initialize Models (Ensure NC=3)
netG = cont_cond_cnn_generator(nz=256).to(device)
netG.load_state_dict(checkpoint["netG_state_dict"])
netG.eval()

# 3. Generate for a specific cell count
# Cell counts are usually normalized 0-1 in CcGAN
target_count = 0.5  # Generate images for a "medium" density
n_images = 16

with torch.no_grad():
    z = torch.randn(n_images, 256).to(device)
    y = torch.full((n_images, 1), target_count).to(device)
    # If your model uses y2h embedding:
    # y = net_y2h(y)

    fake_images = netG(z, y)

    # Rescale from [-1, 1] to [0, 1] for saving
    fake_images = (fake_images + 1) / 2.0
    save_image(fake_images, "fake_cells_count_0_5.png", nrow=4)

print("Generated images saved to fake_cells_count_0.5.png")

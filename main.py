import torch
from pbpunet import PBPUNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# The Vanilla U-Net, referred to as Baseline in the paper
# where all four downsampling layers are conventional MaxPools (kernel=2×2, stride=2) 
lpf_size = None
model_baseline = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)

# A BlurPooled U-Net, referred to as BlurPooling m x m in the paper
# where all four downsampling layers are MaxBlurPools with anti-aliasing filters of 
# the same size (m x m)
lpf_size = [3, 3, 3, 3] # Valid values for m are 2, 3, 4, 5, 6, and 7.
model_blurpooled_3 = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)

# The Pyramidal BlurPooled U-Net (PBPUNet), referred to as PBP in the paper
# where the size of anti-aliasing filters gradually decreases at each downsampling 
# layer from the first (shallow) to the fourth (deep) one.

# Filter sizes used in the paper for the PBP method: 7x7, 5x5, 3x3, 2x2
# Other combinations also can be used, where valid filters sizes are 2, 3, 4, 5, 6, and 7.  
lpf_size = [7, 5, 3, 2]
model_pbp = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)
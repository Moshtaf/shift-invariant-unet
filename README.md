# Shift-Invariant U-Net
Implementation of the networks used in the paper "Investigating Shift-Variance of Convolutional Neural Networks in Ultrasound Image Segmentation."

## Baseline
The vanilla U-Net, referred to as Baseline in the paper, where all four downsampling layers are conventional MaxPools (kernel=2×2, stride=2):
```python
lpf_size = None
model_baseline = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)
```
## BlurPooling
A BlurPooled U-Net, referred to as BlurPooling m x m in the paper, where all four downsampling layers are MaxBlurPools with anti-aliasing filters of the same size (m x m):
```python
lpf_size = [3, 3, 3, 3] # Valid values for m are 2, 3, 4, 5, 6, and 7.
model_blurpooled_3 = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)
```
## Pyramidal BlurPooling
The Pyramidal BlurPooled U-Net (PBPUNet), referred to as PBP in the paper, where the size of anti-aliasing filters gradually decreases at each downsampling layer from the first (shallow) to the fourth (deep) one.
Filter sizes used in the paper for the PBP method: 7x7, 5x5, 3x3, 2x2. Other combinations can also be used, where valid filters sizes are 2, 3, 4, 5, 6, and 7.
```python
lpf_size = [7, 5, 3, 2]
model_pbp = PBPUNet(n_channels=1, n_classes=1, lpf_size=lpf_size).to(device)
```

# Datasets
The list of three publicly available datasets employed in the paper:
* First Ultrasound Dataset: [Breast Ultrasound Dataset B](http://www2.docm.mmu.ac.uk/STAFF/m.yap/dataset.php)
* Second Ultrasound Dataset: [Breast Ultrasound Images Dataset (Dataset BUSI)](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)
* Brain MRI Dataset: [LGG Segmentation Dataset](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)


# Citations
If you found this code useful, please cite the following papers:
* [Sharifzadeh, M., Benali, H. and Rivaz, H., 2022. Investigating Shift-Variance of Convolutional Neural Networks in Ultrasound Image Segmentation. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control.](https://ieeexplore.ieee.org/abstract/document/9743460)
* [Sharifzadeh, M., Benali, H. and Rivaz, H., 2021, September. Shift-Invariant Segmentation in Breast Ultrasound Images. In 2021 IEEE International Ultrasonics Symposium (IUS) (pp. 1-4). IEEE.](https://ieeexplore.ieee.org/abstract/document/9593568)
* [Zhang, R., 2019, May. Making convolutional networks shift-invariant again. In International conference on machine learning (pp. 7324-7334). PMLR.](http://proceedings.mlr.press/v97/zhang19a.html)

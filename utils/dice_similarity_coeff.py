import torch

def dice_similarity_coeff(y, yhat, thresholding=False):
    eps = 0.00001
    dice_th = 0.5
    
    # If thresholding is true, convert the estimated output to a binary image. 
    if (thresholding):
        yhat[yhat > dice_th] = 1
        yhat[yhat <= dice_th] = 0
    
    yflat = y.contiguous().view(-1)
    yhatflat = yhat.contiguous().view(-1)
    
    intersection = torch.abs((yflat * yhatflat)).sum()
    y_sum = torch.sum(yflat)
    yhat_sum = torch.sum(yhatflat)
    
    return 1 - ((2. * intersection + eps) / (y_sum + yhat_sum + eps))
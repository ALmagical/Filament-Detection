import torch
import cv2


def erosion(input:torch.Tensor, ksize, kernel=None , padding=True):

    assert input.dim() == 4

    B, C, H, W = input.shape
    
    if kernel == None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)
    
    kernel = torch.tensor(kernel, dtype=torch.bool)
    k_h, k_w = kernel.shape

    assert k_h == k_w
    if padding:
        pad = (k_h - 1) // 2
        input = torch.nn.functional.pad(input, [pad,pad,pad,pad], mode='constant', value=0)
        out_h = H
        out_w = W
    
    patches = input.unfold(dimension=2, size=k_h, step=1)
    patches = patches.unfold(dimension=3, size=k_h, step=1)

    out, _ = patches[:,:,:,:,kernel].reshape(B, C, out_h, out_w, -1).min(dim=-1)

    return out
    
def medium_filter(input:torch.Tensor, ksize, padding=True):

    assert input.dim() == 4

    B, C, H, W = input.shape
        
    k_h, k_w = ksize

    assert k_h == k_w
    if padding:
        pad = (k_h - 1) // 2
        input = torch.nn.functional.pad(input, [pad,pad,pad,pad], mode='constant', value=0)
        out_h = H
        out_w = W
    
    patches = input.unfold(dimension=2, size=k_h, step=1)
    patches = patches.unfold(dimension=3, size=k_h, step=1)

    mid_indx = round((k_h * k_w)/2)
    patches = patches.reshape(B, C, out_h, out_w, -1)
    patches_sorted, _ = torch.sort(patches, dim=-1)   
    out = patches_sorted[:,:,:,:,mid_indx]

    return out

def mean_filter(input:torch.Tensor, ksize, padding=True):

    assert input.dim() == 4

    B, C, H, W = input.shape
        
    k_h, k_w = ksize

    assert k_h == k_w
    if padding:
        pad = (k_h - 1) // 2
        input = torch.nn.functional.pad(input, [pad,pad,pad,pad], mode='constant', value=0)
        out_h = H
        out_w = W
    
    patches = input.unfold(dimension=2, size=k_h, step=1)
    patches = patches.unfold(dimension=3, size=k_h, step=1)

    out = patches.reshape(B, C, out_h, out_w, -1).mean(dim=-1)

    return out
import lpips
import torch
from basicsr.utils.registry import METRIC_REGISTRY

# Register LPIPS metric in METRIC_REGISTRY
@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Reference: https://arxiv.org/abs/1801.03924

    Args:
        img (ndarray or Tensor): Images with range [0, 255] for ndarray or [0, 1] for Tensor.
        img2 (ndarray or Tensor): Images with range [0, 255] for ndarray or [0, 1] for Tensor.
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: LPIPS result.
    """

    # Ensure both images have the same shape
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Convert input to CHW order if not already
    if input_order == 'HWC':
        img = img.transpose(2, 0, 1)  # Convert to CHW
        img2 = img2.transpose(2, 0, 1)

    # Crop border if needed
    if crop_border != 0:
        img = img[:, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, crop_border:-crop_border, crop_border:-crop_border]

    # Convert to PyTorch Tensor if input is ndarray
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img / 255.).float()
        img2 = torch.from_numpy(img2 / 255.).float()

    # Normalize input to range [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.
    if img2.max() > 1.0:
        img2 = img2 / 255.

    # Ensure input is 4D tensor (Batch, Channel, Height, Width)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    # Load the pretrained LPIPS model (using VGG as default backbone)
    loss_fn = lpips.LPIPS(net='vgg')

    # Calculate LPIPS distance
    lpips_value = loss_fn(img, img2)

    return lpips_value.item()  # Return the LPIPS distance as a float


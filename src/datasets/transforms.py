"""
Data Transformations and Augmentations
Deep Learning AI403 - PUA
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224, is_train: bool = True, use_augmentation: bool = False):
    """
    Returns image transform pipeline and mask transform pipeline.
    For anomaly detection:
      - Normal samples during training do not undergo spatial distortions that simulate defects.
      - Augmentation ablation tests subtle color jitter or rotation vs scratch.
    """
    transform_list = []
    
    # Resize to standard size
    transform_list.append(transforms.Resize((img_size, img_size)))
    
    if is_train and use_augmentation:
        # Mild photometric augmentations for data-strategy ablation
        transform_list.append(transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05))
        
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    img_transform = transforms.Compose(transform_list)
    
    # Mask transform (No normalization, binary 0 or 1)
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return img_transform, mask_transform

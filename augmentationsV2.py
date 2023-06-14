import torch
from typing import Tuple, Union
import torchvision.transforms as T


from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data


class DeviceAgnosticRandomErasing(T.RandomErasing):
 def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
     super().__init__(p=p, scale=scale, ratio=ratio, value=value, inplace=inplace)

 def forward(self, images):
     assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
     B, C, H, W = images.shape
     random_erasing = super(DeviceAgnosticRandomErasing, self).forward
     augmented_images = [random_erasing(img).unsqueeze(0) for img in images]
     augmented_images = torch.cat(augmented_images)
     return augmented_images


class DeviceAgnosticRandomRotation(T.RandomRotation):
 def __init__(self, degrees, interpolation=T.InterpolationMode.NEAREST, expand=False, center=None, fill=0):
     super().__init__(degrees=degrees, interpolation=interpolation, expand=expand, center=center, fill=fill)

 def forward(self, images):
     assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
     B, C, H, W = images.shape
     random_rot = super(DeviceAgnosticRandomRotation, self).forward
     augmented_images = [random_rot(img).unsqueeze(0) for img in images]
     augmented_images = torch.cat(augmented_images)
     return augmented_images
 
 class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images


if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    import matplotlib.patches as patches
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    random_jit = DeviceAgnosticColorJitter(3.1, 0.5, 0.1, 0.1)
    random_er = DeviceAgnosticRandomErasing(p=0.9, scale = (0.05,0.2))
    random_rot = DeviceAgnosticRandomRotation(degrees=20)
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    
    image_paths = [
     "/content/drive/MyDrive/data/@0543037.38@4180974.80@10@S@037.77510@-122.51130@1ooZhwTQyuNt0xiue5YhLg@@134@@@@201311@@.jpg",
     "/content/drive/MyDrive/data/@0543194.44@4181590.30@10@S@037.78064@-122.50948@eCTaveq9DNYVSPZwEAfgAw@@352@@@@201312@@.jpg",
     "/content/drive/MyDrive/data/@0548550.85@4181807.65@10@S@037.78233@-122.44864@whYw-kC1lcXVvBefhF2sIQ@@0@@@@201408@@.jpg",
     "/content/drive/MyDrive/data/@0548602.01@4178059.96@10@S@037.74855@-122.44831@3WgZOVCOEfv5zPJXuYQdMQ@@150@@@@201903@@.jpg"
    ]
    
    trans = T.Compose([
     random_crop,
     random_jit,
     random_er,
     random_rot,
    ])
    
    fig, ax = plt.subplots(len(image_paths), 2, figsize=(5, 2 * len(image_paths)))
    imgs = []
    for i, image_path in enumerate(image_paths):
     # Load the original image
     original_image = Image.open(image_path)
     tensor_image = T.functional.to_tensor(original_image).unsqueeze(0)
     imgs.append(tensor_image)
    images_batch = torch.cat(imgs)
    transformed_images = trans(images_batch)
    
    for i, image_path in enumerate(image_paths):
     augmented_image = T.functional.to_pil_image(transformed_images[i])
     original_image = Image.open(image_path)
    
     # Display the original image
     ax[i, 0].imshow(original_image)
     ax[i, 0].axis('off')
     ax[0, 0].set_title('Original Image', fontsize = 10)
    #         # Display the transformed image
     ax[i, 1].imshow(augmented_image)
     ax[i, 1].axis('off')
     ax[0, 1].set_title('Transformed Image', fontsize = 10)
    
    #         # Draw an arrow pointing from original to transformed image
     arrow_start = [0.2, 0.5]
     arrow_end = [0.8, 0.5]
     arrow = patches.FancyArrowPatch(
         arrow_start, arrow_end,
         arrowstyle='->',
         color='blue',
         linewidth=2,
         connectionstyle='arc3,rad=0.2'
     )
     ax[i, 0].add_patch(arrow)
    
    #     # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0, wspace=0.9)
    
    
    #     # Show the figure
    plt.show()
    
    transformed_images = trans(images_batch)
    
    augmented_batch_ = random_crop(images_batch)
    augmented_batch = random_jit(augmented_batch_)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    bbb = T.functional.to_pil_image(transformed_images[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
    bbb.show()

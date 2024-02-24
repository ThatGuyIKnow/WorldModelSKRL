import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from Utils.TransformerWrapper import Crop

def load_and_display_image(image_path, transform):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply transformations
    transformed_image = transform(image)
    print(transformed_image.shape)
    # Display the image
    plt.imshow(transformed_image.permute(1, 2, 0))  # Convert tensor back to image format
    plt.axis('off')  # Turn off axis
    plt.show(block=True)

if __name__ == "__main__":
    # Example usage:
    image_path = "data/pretrained_carracing-v2/episode_0_step_30.png"  # Replace with your image path

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Grayscale(), 
        transforms.Normalize((0.5,), (0.5,)), 
        Crop(bottom=-50),
        transforms.Resize((64, 64)),
        ])

    load_and_display_image(image_path, transform)

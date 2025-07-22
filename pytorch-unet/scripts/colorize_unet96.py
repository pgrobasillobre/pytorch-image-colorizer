import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

# Add path to import model from custom module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classes')))
from model_unet96 import UNetColorization96

# ----------------------------
# Function: Load and preprocess grayscale image
# ----------------------------
def load_and_preprocess_image(img_path):
    """
    Load a grayscale image, resize to 96x96, convert to tensor, and add batch dimension.

    Args:
        img_path (str): Path to the input image.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 1, 96, 96)
    """
    img = Image.open(img_path).convert("L").resize((96, 96))  # Convert to grayscale and resize to 96x96
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)      # Shape: (1, 1, 96, 96)
    return img_tensor

# ----------------------------
# Function: Save output RGB image from model prediction
# ----------------------------
def save_output_image(tensor, output_path):
    """
    Convert model output tensor to image and save it to disk.

    Args:
        tensor (torch.Tensor): Model output tensor with shape (1, 3, H, W)
        output_path (str): Path to save the colorized image
    """
    output_img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Shape: (H, W, 3)
    output_img = (output_img * 255).astype(np.uint8)                        # Rescale to 0–255 and convert to uint8
    Image.fromarray(output_img).save(output_path)
    print(f"Colorized image saved to {output_path}")

# ----------------------------
# Main function: Run the prediction pipeline
# ----------------------------
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to grayscale input image")
    parser.add_argument("--output", required=True, help="Path to save colorized output")
    parser.add_argument("--model", default="../model/colorizer_training_history_unet96.pkl",
                        help="Path to trained U-Net model weights")
    args = parser.parse_args()

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained U-Net model
    model = UNetColorization96().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()  # Set model to evaluation mode

    # Load and preprocess the input grayscale image
    input_tensor = load_and_preprocess_image(args.input).to(device)

    # Run inference (no gradient calculation needed)
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Save the colorized output image
    save_output_image(output_tensor, args.output)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from model_unet import UNetColorization

# Preprocess grayscale input image
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("L").resize((32, 32))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Shape: (1, 1, 32, 32)
    return img_tensor

# Save output image from model prediction
def save_output_image(tensor, output_path):
    output_img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_img = (output_img * 255).astype(np.uint8)
    Image.fromarray(output_img).save(output_path)
    print(f"Colorized image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to grayscale input image")
    parser.add_argument("--output", required=True, help="Path to save colorized output")
    parser.add_argument("--model", default="pytorch/models/colorizer_model_unet.pth",
                        help="Path to trained U-Net model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNetColorization().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load and preprocess image
    input_tensor = load_and_preprocess_image(args.input).to(device)

    # Predict
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Save result
    save_output_image(output_tensor, args.output)

if __name__ == "__main__":
    main()

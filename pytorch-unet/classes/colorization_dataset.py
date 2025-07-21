# Custom PyTorch dataset for grayscale-to-color image colorization using STL-10
from torch.utils.data import Dataset
from torchvision.datasets import STL10
from torchvision import transforms
import torchvision.transforms.functional as TF

class STL10Colorization(Dataset):
    def __init__(self, train=True):
        """
        Initializes the STL10 dataset for colorization.
        Converts images to tensors and prepares grayscale inputs with RGB targets.

        Args:
            train (bool): If True, uses the training split; otherwise, uses the test split.
        """
        # Select split based on training or testing
        split = "train" if train else "test"

        # Load the STL10 dataset (96x96 RGB images) and convert images to PyTorch tensors
        self.data = STL10(
            root="./data",
            split=split,
            download=True,
            transform=transforms.ToTensor()  # Converts images to shape (3, 96, 96)
        )

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample: grayscale input and RGB target.

        Args:
            idx (int): Index of the sample.

        Returns:
            gray_img (Tensor): Grayscale image of shape (1, 96, 96).
            color_img (Tensor): Original RGB image of shape (3, 96, 96).
        """
        color_img, _ = self.data[idx]  # Get RGB image, ignore label
        gray_img = TF.rgb_to_grayscale(color_img)  # Convert to grayscale
        return gray_img, color_img

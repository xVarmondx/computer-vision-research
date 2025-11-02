import os
from glob import glob
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_metadata(file_path):
    """
    Extracts class name and image type based on file path and name conventions.
    """
    class_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path).lower()

    if "stamp" in file_name:
        image_type = "Stamp (Photo)"
    elif "digit" in file_name:
        image_type = "Hand-Drawn (Scanned/Digitalized)"
    elif "drawn" in file_name:
        image_type = "Hand-Drawn (Photo)"
    else:
        image_type = "Unknown"

    return class_name, image_type


def load_dataframe(data_path):
    """
    Scans the data directory and creates the main DataFrame containing metadata.
    """
    all_files = glob(os.path.join(data_path, "*", "*.jpg"))
    if not all_files:
        print(f"WARNING: No .jpg files found in {data_path}")
        return pd.DataFrame(columns=["class_name", "image_type", "filepath"])

    data = [get_metadata(i) + (i,) for i in all_files]
    data_df = pd.DataFrame(data, columns=["class_name", "image_type", "filepath"])
    return data_df


class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading images and numerical labels.
    """

    def __init__(self, df, class_to_idx, transform=None):
        self.df = df
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.labels = self.df['class_name'].map(self.class_to_idx).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['filepath'].iloc[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_subset_loader(df_subset, transform, class_to_idx, batch_size, shuffle=False):
    """
    Creates a DataLoader instance for a given subset DataFrame.
    """
    dataset = ImageDataset(df_subset, class_to_idx, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(data_df, class_to_idx, batch_size):
    """
    Splits the data into train, validation, and test sets and creates DataLoaders.

    Returns: train_loader, val_loader, test_loader, test_df
    """


    train_val_df, test_df = train_test_split(
        data_df, test_size=0.1, random_state=42, stratify=data_df['class_name']
    )


    train_df, val_df = train_test_split(
        train_val_df, test_size=(0.1 / 0.9), random_state=42, stratify=train_val_df['class_name']
    )


    train_transforms = transforms.Compose([
        transforms.Resize((130, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((130, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = get_subset_loader(train_df, train_transforms, class_to_idx, batch_size, shuffle=True)
    val_loader = get_subset_loader(val_df, val_test_transforms, class_to_idx, batch_size)
    test_loader = get_subset_loader(test_df, val_test_transforms, class_to_idx, batch_size)

    return train_loader, val_loader, test_loader, test_df
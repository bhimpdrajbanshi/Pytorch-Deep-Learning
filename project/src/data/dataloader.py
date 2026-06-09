from torch.utils.data import DataLoader, random_split

from src.data.dataset import ImageClassificationDataset


def get_dataloader(
    root_dir,
    transform,
    batch_size=4,
    train_ratio=0.8,
    shuffle=True
):

    # Load complete dataset
    dataset = ImageClassificationDataset(
        root_dir=root_dir,
        transform=transform
    )

    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        dataset.classes
    )
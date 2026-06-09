from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from src.data.dataset import ImageClassificationDataset
from src.data.dataloader import get_dataloader
from src.models.baseline_model import SimpleCNN
from src.training.trainer import Trainer


def main():

    # ------------------------
    # 1. Project Paths
    # ------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent

    data_dir = PROJECT_ROOT / "dataset" / "flowers"

    print("Dataset path:", data_dir)

    # ------------------------
    # 2. Transform
    # ------------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # ------------------------
    # 3. Load Full Dataset
    # ------------------------
    full_dataset = ImageClassificationDataset(
        root_dir=str(data_dir),
        transform=transform
    )

    print("Total samples:", len(full_dataset))
    print("Classes:", full_dataset.classes)

    # ------------------------
    # 4. Split Train & Validation
    # ------------------------
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size]
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # ------------------------
    # 5. DataLoader
    # ------------------------
    train_loader, val_loader, classes = get_dataloader(
        root_dir=str(data_dir),
        transform=transform,
        batch_size=4
    )


    # ------------------------
    # 6. Model
    # ------------------------
    model = SimpleCNN(
        num_classes=len(full_dataset.classes)
    )

    print(model)

    # ------------------------
    # 7. Trainer
    # ------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.001,
        device="cpu"   # change to cuda if available
    )

    # ------------------------
    # 8. Training
    # ------------------------
    num_epochs = 5

    for epoch in range(num_epochs):

        train_loss, train_acc = trainer.train_one_epoch()
        val_acc = trainer.evaluate()

        print("\n" + "=" * 40)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("=" * 40)
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.4f}")
        print(f"Val Acc    : {val_acc:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    main()

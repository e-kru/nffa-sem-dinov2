from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


DATASET_ROOT = "/mnt/data/stud-ueqio/datasets/nffa_sem_100"


def get_dataset():
    """
    Erstellt das PyTorch-Dataset.
    """

    transform = transforms.ToTensor()

    dataset = datasets.ImageFolder(
        root=DATASET_ROOT,
        transform=transform
    )

    return dataset


def get_dataloader(batch_size=4, shuffle=True):
    """
    Erstellt einen DataLoader für das Dataset.
    """

    dataset = get_dataset()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader


def show_batch(images, labels, classes):
    """
    Zeigt alle Bilder eines Batches zusammen mit ihren Klassen an.
    """

    for i in range(len(images)):
        image = images[i]

        # PyTorch: [C, H, W]
        # matplotlib: [H, W, C]
        image = image.permute(1, 2, 0)

        plt.figure()
        plt.imshow(image)
        plt.title(classes[labels[i]])
        plt.axis("off")
        plt.show()


if __name__ == "__main__":

    # Dataset erstellen
    dataset = get_dataset()

    print("Number of images:", len(dataset))
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)

    # Einzelnes Beispiel laden
    image, label = dataset[0]

    print("Single image shape:", image.shape)
    print("Single label:", label)
    print("Single class:", dataset.classes[label])

    # DataLoader erstellen
    loader = get_dataloader(
        batch_size=4,
        shuffle=True
    )

    # Einen Batch laden
    images, labels = next(iter(loader))

    print("Batch shape:", images.shape)
    print("Labels:", labels)

    # Batch visualisieren
    show_batch(
        images,
        labels,
        dataset.classes
    )
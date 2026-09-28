from torchvision import datasets, transforms
from torch.utils.data import DataLoader


DATASET_ROOT = "/mnt/data/stud-ueqio/datasets/nffa_sem_100"

transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    root = DATASET_ROOT,
    transform = transform
)

image, label = dataset[0]

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

print("Image shape:", image.shape)
print("Label:", label)
print("Class name:", dataset.classes[label])

loader = DataLoader



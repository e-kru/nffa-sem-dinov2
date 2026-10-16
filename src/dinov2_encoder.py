import torch
from torchvision import transforms

from src.dataset import get_dataset, get_dataloader
from src.preprocessing_v2 import crop_metadata_strip




MODEL_NAME = "dinov2_vits14"


def get_dinov2_transform():
    transform = transforms.Compose([
        transforms.Lambda(crop_metadata_strip),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])
    return transform


def get_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model

if __name__ == "__main__":
    model = get_dinov2()
    transform = get_dinov2_transform()

    loader = get_dataloader(
    batch_size=4,
    shuffle=True,
    transform=transform
    )

    images, labels = next(iter(loader))

    print("Input batch shape:", images.shape)
    print("Labels:", labels)

    with torch.no_grad():
        features = model(images)

    print("Feature batch shape:", features.shape)
        

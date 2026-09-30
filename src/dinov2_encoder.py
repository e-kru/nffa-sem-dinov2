import torch
from torchvision import transforms
from dataset import get_dataset
from dataset import get_dataloader



MODEL_NAME = "dinov2_vits14"


def get_dinov2_transform():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.458, 0.448, 0.408),
            std=(0.268, 0.261, 0.275)
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
        

import torch
from dataset import get_dataloader
from dinov2_encoder import get_dinov2, get_dinov2_transform

def extract_features(model, loader, max_batches=None):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            features = model(images)
            all_features.append(features)
            all_labels.append(labels)

            print(
                "Batch:", batch_idx,
                "Images", images.shape,
                "Features", features.shape
                   )

            if max_batches is not None and batch_idx +1 >= max_batches:
                break
        all_features = torch.cat(all_features,dim =0)
        all_labels = torch.cat(all_labels,dim =0)
        
        return all_features, all_labels


if __name__ == "__main__":
    transform = get_dinov2_transform()
    loader = get_dataloader(
        batch_size=4,
        shuffle=False,
        transform = transform
    )
    model = get_dinov2()
    features, labels = extract_features(model, loader, max_batches=2)

    print("Final feature shape:", features.shape)
    print("Final label shape:" , labels.shape)
    print("Lables", labels)


import torch

MODEL_NAME = "dinov2_vits14"

def get_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model

if __name__ == "__main__":
    model = get_dinov2()
    print("Training mode:", model.training)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
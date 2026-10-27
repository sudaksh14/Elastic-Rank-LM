# scripts/extract_cifar10_features.py
import torch, torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import argparse

# Add this just before `if __name__ == '__main__':`
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='~/cifar_feats',
                        help='Path to save the extracted feature tensors')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def get_backbone(device='cuda'):
    resnet = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1)
    # Throw away the classifier – keep global-avg-pooled 512-D vector
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval().to(device).requires_grad_(False)
    return backbone

def save_split(split, dataloader, backbone, out_dir):
    feats, labels = [], []
    with torch.no_grad():
        for imgs, y in tqdm(dataloader, desc=f'{split}'):
            imgs = imgs.to(backbone[0].weight.device)
            f = backbone(imgs).flatten(1).cpu()   # shape [B,512]
            feats.append(f)
            labels.append(y)
    torch.save(
        {
            'feats': torch.cat(feats),   # [N,512]
            'labels': torch.cat(labels)  # [N]
        },
        out_dir / f'{split}.pt'
    )

def main(root='~/cifar_feats', batch_size=256, workers=4):
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    # Normal CIFAR-10 datamodule-style transforms
    transform = tv.transforms.Compose([
        tv.transforms.Resize(224),                   # ResNet expects 224×224
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485,0.456,0.406],
                                std=[0.229,0.224,0.225])
    ])

    trainset = tv.datasets.CIFAR10(root='~/data', train=True,
                                   download=True, transform=transform)
    testset  = tv.datasets.CIFAR10(root='~/data', train=False,
                                   download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)
    testloader  = DataLoader(testset , batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True)

    backbone = get_backbone()

    save_split('train', trainloader, backbone, root)
    save_split('test',  testloader,  backbone, root)

if __name__ == '__main__':
    args = parse_args()
    main(root=args.root, batch_size=args.batch_size, workers=args.workers)
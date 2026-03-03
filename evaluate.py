import os
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the dataset")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training",
    )
    

    parser.add_argument('--model_path', type=str, default='./models/model.pth', help='Path to save or load models')
    parser.add_argument('--train', type='store_action', help='Whether to train the model or evaluate')
    parser.add_argument(
        '--arch', type=str, default='./models', choices=[
            'lenet5', 'mini_vgg',
        ],
        help='Path to save or load models')
    parser.add_argument('--trigger_patern', type=str, default='lenet', help='Which backdoor model to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--target_label', type=int, default=0, help='Target label for the backdoor attack')
    args = parser.parse_args()
    
    return args



class MiniVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # 128 channels × 7 × 7 feature map
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))              # (B, 32, 28, 28)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B, 64, 14, 14)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  # (B, 128, 7, 7)

        x = x.flatten(1)                       # (B, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_data(args):
    # Transforms: tensor + normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = datasets.MNIST(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=args.data_path,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    if args.arch == 'lenet5':
        model = LeNet5()
    elif args.arch == 'mini_vgg':
        model = MiniVGG()
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    train_loader, test_loader = get_data(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)


    acc = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {acc:.2f}%")
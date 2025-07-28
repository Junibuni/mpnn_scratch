import torch
import argparse
from dataloader import load_data
from train import train
from test import test

from pyg_model import GNN as PyGMPNN
from scratch_model import ScratchMPNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pyg', choices=['pyg', 'scratch'],
                        help='Choose which model to use: pyg or scratch')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, dataset = load_data()

    if args.model == 'pyg':
        model = PyGMPNN(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes
        )
    elif args.model == 'scratch':
        model = ScratchMPNN(
            in_channels=dataset.num_node_features,
            hidden_channels=64,
            num_classes=dataset.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 31):
        loss = train(model, train_loader, optimizer, device)
        acc = test(model, test_loader, device)
        print(f"[{args.model.upper()}] Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

if __name__ == '__main__':
    main()
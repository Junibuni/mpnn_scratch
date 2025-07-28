import torch
from pyg_model import GNN
from dataloader import load_data
from train import train
from test import test

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, dataset = load_data()

    model = GNN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        num_classes=dataset.num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 31):
        loss = train(model, train_loader, optimizer, device)
        acc = test(model, test_loader, device)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

if __name__ == '__main__':
    main()

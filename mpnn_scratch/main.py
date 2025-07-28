import torch
import argparse
from dataloader import load_data
from train import train
from test import test
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pyg_model import GNN as PyGMPNN
from scratch_model import ScratchMPNN

@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds = []
    labels = []
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1).cpu()
        label = data.y.cpu()
        preds.append(pred)
        labels.append(label)
    return torch.cat(preds), torch.cat(labels)

def main():
    EPOCHS = 1000
    VERBOSE = False
    
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, dataset = load_data()
    edge_dim = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 0

    pyg_model = PyGMPNN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        num_classes=dataset.num_classes,
        edge_dim=edge_dim
    ).to(device)

    scratch_model = ScratchMPNN(
        in_channels=dataset.num_node_features,
        hidden_channels=64,
        num_classes=dataset.num_classes,
        edge_dim=edge_dim
    ).to(device)

    pyg_optimizer = torch.optim.AdamW(pyg_model.parameters(), lr=0.01)
    scratch_optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=0.01)

    pyg_losses, scratch_losses = [], []
    pyg_accs, scratch_accs = [], []
    
    for epoch in range(1, EPOCHS):
        pyg_loss = train(pyg_model, train_loader, pyg_optimizer, device)
        pyg_acc = test(pyg_model, test_loader, device)

        pyg_losses.append(pyg_loss)
        pyg_accs.append(pyg_acc)

        if VERBOSE:
            print(f"[PYG]     Epoch {epoch:02d}, Loss: {pyg_loss:.4f}, Test Acc: {pyg_acc:.4f}")
        
    for epoch in range(1, EPOCHS):
        scratch_loss = train(scratch_model, train_loader, scratch_optimizer, device)
        scratch_acc = test(scratch_model, test_loader, device)

        scratch_losses.append(scratch_loss)
        scratch_accs.append(scratch_acc)

        if VERBOSE:
            print(f"[SCRATCH] Epoch {epoch:02d}, Loss: {scratch_loss:.4f}, Test Acc: {scratch_acc:.4f}")

    plt.figure()
    plt.plot(pyg_losses, label='PyG')
    plt.plot(scratch_losses, label='Scratch')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_comparison.png")
    plt.close()

    pyg_preds, pyg_labels = get_predictions(pyg_model, test_loader, device)
    scratch_preds, scratch_labels = get_predictions(scratch_model, test_loader, device)

    cm = confusion_matrix(pyg_labels, pyg_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("PyG Confusion Matrix")
    plt.savefig("pyg_confusion_matrix.png")
    
    cm = confusion_matrix(scratch_labels, scratch_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Scratch Confusion Matrix")
    plt.savefig("scratch_confusion_matrix.png")


    print("Plots saved: loss_comparison.png, parity_plot.png")

if __name__ == '__main__':
    main()
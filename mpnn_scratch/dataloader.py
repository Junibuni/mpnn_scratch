from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def load_data(batch_size=32):
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, dataset

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.c_model import Classifier
from src.c_dataset import ChickenDataset

if __name__ == "__main__":
    main_tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_set = ChickenDataset("./train_data.csv", tfm=main_tfm)
    data_map = data_set.get_label_map()

    split_ratio = 0.8

    train_size = int(split_ratio * len(data_set))
    valid_size = len(data_set) - train_size
    print("Train size:", train_size, "Valid size:", valid_size)

    train_set, valid_set = torch.utils.data.random_split(data_set, [train_size, valid_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 8
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    best_acc = 0
    for epoch in range(n_epochs):
        model.train()
        for batch in tqdm(train_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        total = 0
        correct = 0
        for batch in tqdm(valid_loader):
            imgs, labels = batch
            logits = model(imgs.to(device))
            _, pred = torch.max(logits, 1)
            total += len(labels)
            correct += (pred == labels.to(device)).sum()
        print(f"Epoch {epoch+1}: {correct}/{total} = {correct/total}")
        if correct/total > best_acc:
            best_acc = correct/total
            torch.save(model.state_dict(), f"./models/model{best_acc:.4f}.pth")
        



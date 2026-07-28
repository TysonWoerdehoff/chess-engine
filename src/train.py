import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import load_dataframe, ChessDataset
from model import ChessNet

def main():
    df = load_dataframe("data/chessData.csv", n_rows = 1_000_000)
    dataset = ChessDataset(df)
    n = len(dataset)
    train_size = int(n * .9)
    val_size = (n - train_size)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    print("train batches:", len(train_loader), "val batches:", len(val_loader))
    
    model = ChessNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_history = []
    val_history = []
    best_val = float("inf")

    for epoch in range(8):
        model.train()
        running = 0.0
        
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x,y in val_loader:
                pred = model(x)
                loss = criterion(pred, y)
                val_running += loss.item()
                
        val_avg = val_running / len(val_loader)
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), "models/chess_net.pt")
        avg = running / len(train_loader)
        
        print("epoch", epoch, "train:", avg, "val:", val_avg)
        train_history.append(avg)
        val_history.append(val_avg)
    
    
    plt.plot(train_history, label="train")
    plt.plot(val_history, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.savefig("loss_curve.png")    
    print("saved model")
    
if __name__ == "__main__":
    main()
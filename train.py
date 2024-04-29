import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from dataset import RoadSeg
from model import Unet

generator = torch.Generator().manual_seed(44)
device = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_IMG = "B:/MLStuff/learning/datasets/segmentation/road_seg/1-Road-Ortho/"
ROOT_MASK = "B:/MLStuff/learning/datasets/segmentation/road_seg/1-Road-Masks"
MODEL_PATH = "B:/MLStuff/learning/Unet/models"
train_dataset = RoadSeg(ROOT_IMG, ROOT_MASK)
train_final, _ = random_split(train_dataset, [0.3, 0.7], generator=generator)
test_dataset = RoadSeg(ROOT_IMG, ROOT_MASK, train=False)
EPOCHS = 10
BATCH_SIZE = 40
LR = 3e-4
# TO use full dataset for training put train_dataset in the trainLoader
trainLoader = DataLoader(dataset=train_final, batch_size=BATCH_SIZE, shuffle=True)
testLoader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
criterion = nn.BCEWithLogitsLoss()
model = Unet(3, 1).to(device)
optimizer = Adam(model.parameters(), lr=LR)


def main():
    train_loss = []
    valid_loss = []
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train = 0
        for img, mask in tqdm(trainLoader):
            optimizer.zero_grad()
            img, mask = img.to(device), mask.to(device)
            y_hat = model(img)
            loss = criterion(y_hat, mask)
            train += loss.item()
            loss.backward()
            optimizer.step()
        train = train / len(trainLoader)
        train_loss.append(train)

        valid = 0
        model.eval()
        with torch.no_grad():
            for img, mask in tqdm(testLoader):
                img, mask = img.to(device), mask.to(device)
                y_hat = model(img)
                loss = criterion(y_hat, mask)
                valid += loss.item()
            valid = valid / len(testLoader)
            valid_loss.append(valid)
        print("-"*30)
        print(f"Training Loss EPOCH {epoch + 1}: {train:.4f}")
        print(f"Validation Loss EPOCH {epoch + 1}: {valid:.4f}")
        print("-"*30)
    
    torch.save(model.state_dict(), f"{MODEL_PATH}/model1.pt")
    plot_and_save(train_loss, valid_loss)


def plot_and_save(train_loss, valid_loss):
    sns.lineplot(x=range(1, EPOCHS + 1), y=train_loss, label='Training Loss', linewidth=2)
    sns.lineplot(x=range(1, EPOCHS + 1), y=valid_loss, label='Validation Loss', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{MODEL_PATH}/loss_plot.png")


if __name__ == '__main__':
    main()
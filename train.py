import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from get_loader import get_loader
from model import CNNtoRNN
from utils import save_checkpoint

def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(dataset.vocab), num_layers=1).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Tensorboard setup
    writer = SummaryWriter("runs/flickr")
    step = 0

    model.train()

    for epoch in range(100):  # You can adjust number of epochs
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            loss.backward()
            optimizer.step()

            writer.add_scalar("Training Loss", loss.item(), global_step=step)
            step += 1

        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename="flickr8k_model.pth")

if __name__ == "__main__":
    train()

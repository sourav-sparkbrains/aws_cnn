import os
import torch
import mlflow
import argparse
import torchvision.transforms as transforms

from torch import nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, classification_report

from src.model import XRAYNet
from src.config import settings
from src.dataset import XRAYDataset

load_dotenv()

def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data_path = os.getenv('SM_CHANNEL_TRAIN', str(settings.RAW_DATA_DIR))
    return XRAYDataset(data_path,transform)

def get_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0

        model.train()

        for image, label in train_loader:
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (label == output.argmax(dim=1)).float().mean()

        train_loss = train_loss / len(train_loader)
        train_accuracy = train_accuracy / len(train_loader)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

        print(f"Epoch {epoch}: Loss {train_loss:.4f} and Accuracy {train_accuracy:.4f}")

def evaluate_model(model, test_loader, device):
    all_preds, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            probs = model(image)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(label.cpu())

    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()

    f1 = f1_score(labels_np, preds_np, average='macro')
    print(f"F1: {f1:.4f}")
    mlflow.log_metric("f1_score", f1)
    print(classification_report(labels_np, preds_np))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=settings.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=settings.BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=settings.LEARNING_RATE)
    args = parser.parse_args()

    dataset = load_data()

    train_size = int(len(dataset) * settings.TRAIN_SIZE)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader, test_loader = get_dataloader(train_dataset, test_dataset, batch_size=args.batch_size)

    mlflow.set_experiment(settings.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=settings.RUN_NAME):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = XRAYNet()
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        epochs = args.epochs

        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)

        train_model(model, train_loader, criterion, optimizer, epochs, device)
        evaluate_model(model, test_loader, device)
        mlflow.pytorch.log_model(model, "model")
        model_dir = os.getenv('SM_MODEL_DIR', '.')
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))

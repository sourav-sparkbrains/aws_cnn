import os
import torch
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from torchvision import transforms

from src.model import XRAYNet
from src.config import settings

load_dotenv()

model = XRAYNet()
model.load_state_dict(torch.load(f'{os.getenv('SM_MODEL_DIR')}/model.pt', weights_only=True))
model.eval()

data_path = os.getenv('SM_CHANNEL_TRAIN', str(settings.RAW_DATA_DIR))
classes = sorted([d.name for d in Path(data_path).iterdir() if d.is_dir()])
LABELS = {i: cls for i, cls in enumerate(classes)}

def pre_process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = transform(Image.open(image_path)).unsqueeze(0)
    return image


def get_xray_prediction(image_path):
    image = pre_process_image(image_path)

    with torch.no_grad():
        probs = model(image)
        preds = probs.argmax(dim=1)

    return LABELS[preds.item()]


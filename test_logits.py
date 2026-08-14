import torch
from training.train_violence_pytorch import ViolenceCNN_LSTM
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViolenceCNN_LSTM().to(device)
model.load_state_dict(torch.load(config.VIOLENCE_MODEL_PATH, map_location=device))
model.eval()

# Dummy input representing 16 frames of 3 channels 224x224
dummy_input = torch.randn(1, 16, 3, 224, 224).to(device)
with torch.no_grad():
    res = model(dummy_input)
    print("Raw output (logits):", res.item())
    print("Sigmoid probability:", torch.sigmoid(res).item())

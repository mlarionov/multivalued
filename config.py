import torch

# Shared constants and configuration for the multivalued RNN project
START_TOKEN = -1.0
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100
MODEL_PATH = "multivalued_rnn.pth"


# Device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

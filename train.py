import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_utils import generate_dataset
from model import MultivaluedRNN
from config import START_TOKEN, HIDDEN_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATH, DEVICE
import numpy as np

class MultivaluedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_k = max(len(y) for _, y in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y_list = self.data[idx]
        K = len(y_list)
        
        # Pad y_targets and stop_targets
        y_padded = np.zeros(self.max_k)
        stop_padded = np.zeros(self.max_k)
        mask = np.zeros(self.max_k)
        
        y_padded[:K] = y_list
        stop_padded[:K-1] = 0.0
        stop_padded[K-1] = 1.0
        mask[:K] = 1.0
        
        return (
            torch.tensor([x], dtype=torch.float32),
            torch.tensor(y_padded, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(stop_padded, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(-1)
        )

def train():
    raw_data = generate_dataset(2000)
    dataset = MultivaluedDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MultivaluedRNN(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss(reduction='none')
    bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for x_batch, y_targets, stop_targets, mask in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_targets = y_targets.to(DEVICE)
            stop_targets = stop_targets.to(DEVICE)
            mask = mask.to(DEVICE)
            
            optimizer.zero_grad()
            B = x_batch.shape[0]
            max_k = y_targets.shape[1]
            
            h = model.init_hidden(x_batch)
            y_prev = torch.full((B, 1), START_TOKEN, dtype=torch.float32, device=DEVICE)
            
            batch_loss = 0
            for i in range(max_k):
                y_pred, stop_logit, h = model(x_batch, y_prev, h)
                
                # Apply mask to losses
                step_mask = mask[:, i]
                loss_y = (mse_loss(y_pred, y_targets[:, i]) * step_mask).sum()
                loss_stop = (bce_loss(stop_logit, stop_targets[:, i]) * step_mask).sum()
                
                batch_loss += (loss_y + 0.1 * loss_stop)
                
                # Teacher forcing: using ground truth y_prev for next step
                y_prev = y_targets[:, i]
            
            # Normalize by active elements in batch
            batch_loss = batch_loss / mask.sum()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()

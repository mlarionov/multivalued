import torch
from model import MultivaluedRNN
from data_utils import generate_dataset
from config import START_TOKEN, HIDDEN_SIZE, MODEL_PATH, DEVICE
import matplotlib.pyplot as plt
import numpy as np

def generate_sequences_batch(model, x_batch, max_steps=10):
    """Vectorized generation of y values for a batch of x."""
    x_batch = x_batch.to(DEVICE)
    B = x_batch.shape[0]
    with torch.no_grad():
        h = model.init_hidden(x_batch)
        y_prev = torch.full((B, 1), START_TOKEN, dtype=torch.float32, device=DEVICE)

        active_mask = torch.ones(B, dtype=torch.bool, device=DEVICE)
        batch_results = [[] for _ in range(B)]

        for _ in range(max_steps):
            y_pred, stop_logit, h = model(x_batch, y_prev, h)
            stop_probs = torch.sigmoid(stop_logit).squeeze(-1)

            # Update results for active samples
            # Move data back to CPU for list storage
            stop_probs_cpu = stop_probs.cpu()
            y_pred_cpu = y_pred.cpu()
            active_mask_cpu = active_mask.cpu()

            for i in range(B):
                if active_mask_cpu[i]:
                    batch_results[i].append(y_pred_cpu[i].item())
                    if stop_probs_cpu[i] > 0.5:
                        active_mask[i] = False

            if not active_mask.any():
                break
            y_prev = y_pred

    return batch_results

def get_predictions(model, x_range):
    """Batch processes a range of x values using vectorized sampling."""
    x_tensor = torch.tensor(x_range, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    batch_results = generate_sequences_batch(model, x_tensor)
    
    pred_xs = []
    pred_ys = []
    for x_val, y_list in zip(x_range, batch_results):
        for y in y_list:
            pred_xs.append(x_val)
            pred_ys.append(y)
    return pred_xs, pred_ys

def plot_results(gt_xs, gt_ys, pred_xs, pred_ys, save_path='prediction_results.png'):
    """Generates and saves the comparison plot."""
    plt.figure(figsize=(10, 8))
    plt.scatter(gt_xs, gt_ys, c='gray', s=1, alpha=0.3, label='Ground Truth')
    plt.scatter(pred_xs, pred_ys, c='red', s=5, alpha=0.6, label='RNN Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RNN-based Multivalued Function Prediction')
    plt.legend()
    plt.savefig(save_path)
    print(f"Prediction results saved to {save_path}")

def predict():
    # Load model
    model = MultivaluedRNN(hidden_size=HIDDEN_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Generate predictions
    x_test = np.linspace(-0.1, 1.1, 200)
    pred_xs, pred_ys = get_predictions(model, x_test)

    # Get ground truth
    dataset_gt = generate_dataset(500)
    gt_xs = []
    gt_ys = []
    for x, y_list in dataset_gt:
        for y in y_list:
            gt_xs.append(x)
            gt_ys.append(y)

    # Plot
    plot_results(gt_xs, gt_ys, pred_xs, pred_ys)

if __name__ == "__main__":
    predict()

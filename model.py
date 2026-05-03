import torch
import torch.nn as nn

class MultivaluedRNN(nn.Module):
    def __init__(self, hidden_size=128):
        super(MultivaluedRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # More expressive initial hidden state network
        self.init_h = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # RNN layer
        self.rnn = nn.GRUCell(2, hidden_size)
        
        # Deeper Output Heads
        self.fc_y = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc_stop = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x, y_prev, h):
        rnn_input = torch.cat([x, y_prev], dim=1)
        h_next = self.rnn(rnn_input, h)
        
        y_pred = self.fc_y(h_next)
        stop_logit = self.fc_stop(h_next)
        
        return y_pred, stop_logit, h_next

    def init_hidden(self, x):
        return self.init_h(x)

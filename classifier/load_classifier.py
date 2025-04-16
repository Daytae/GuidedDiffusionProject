import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd

class EsmClassificationHead(nn.Module):
    # slightly modified from the original ESM classification head
    def __init__(self, input_dim=256):
        super().__init__()
        self.dense = nn.Linear(input_dim, 2048)
        self.dropout = nn.Dropout(0)
        self.dense2 = nn.Linear(2048, 2048)
        self.dense3 = nn.Linear(2048, 2048)
        self.out_proj = nn.Linear(2048, 2)
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.dense3(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
def load_model(model_path='classifier/best_model.pt'):
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load saved model data
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with saved input dimension
    model = EsmClassificationHead(input_dim=256).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, device

# def predict(model, embeddings, device, batch_size=100):
    
#     # Convert to torch tensor
#     embeddings = torch.FloatTensor(embeddings)
#     predictions = []
#     all_logits = []
#     all_probs = []
    
#     with torch.inference_mode():
        
#         for i in range(0, len(embeddings), batch_size):
#             batch = embeddings[i:i + batch_size].to(device)
#             logits = model(batch)
#             probs = torch.softmax(logits, dim=1)
#             preds = torch.argmax(logits, dim=1)
            
#             all_logits.extend(logits.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
#             predictions.extend(preds.cpu().numpy())
    
#     return (np.array(predictions), 
#             np.array(all_logits), 
#             np.array(all_probs))

def predict(model, embeddings, device):

    # Convert to torch tensor
    model.eval()
    logits = model(embeddings)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)
    return preds, logits, probs
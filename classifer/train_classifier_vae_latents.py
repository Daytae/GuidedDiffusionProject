import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PeptideDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)  # Changed to LongTensor for CrossEntropyLoss
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class EsmClassificationHead(nn.Module):
    # slightly modified from the original ESM classification head
    def __init__(self, input_dim=256):
        super().__init__()
        self.dense = nn.Linear(input_dim, 2048)
        self.dropout = nn.Dropout(0.05)
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

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for embeddings, labels in tqdm(train_loader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.inference_mode():
        for embeddings, labels in tqdm(data_loader, desc="Evaluating"):
            embeddings = embeddings.to(device)
            logits = model(embeddings)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    return {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions)
    }

def main():
    # Load data
    print("Loading training data...")

    train_df = pd.read_csv('./EP_aho_corasick_labels_exclude_active.csv')  # Assuming columns: 'text' and 'labels'
    train_embeddings = np.load('./EP_aho_corasick_labels_exclude_active_latents.npy')  # 256-dim embeddings

    # Load test data (original full dataset)
    print("Loading test data...")
    test_df = pd.read_csv('test_peptides.csv')  # Original file with 'text' and 'labels'
    test_embeddings = np.load('test_peptides_latents.npy')  # Original embeddings
    
    # Create datasets
    train_dataset = PeptideDataset(train_embeddings, train_df['labels'].values)
    test_dataset = PeptideDataset(test_embeddings, test_df['labels'].values)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model, criterion, and optimizer
    model = EsmClassificationHead().to(device)
    criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for 2-class output
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    num_epochs = 10
    best_accuracy = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        metrics = evaluate(model, test_loader, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            torch.save(model.state_dict(), 'best_model.pt')
    
    # Save final model and training info
    final_save = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_metrics': metrics,
        'best_accuracy': best_accuracy,
        'input_dim': 256
    }
    torch.save(final_save, 'final_model.pt')
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
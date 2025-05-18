import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model for gait classification")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="results/gait_features_flat_merged.npy",
                        help="Path to input data file")
    parser.add_argument("--model_save_path", type=str, default="results/best_model.pt",
                        help="Path to save the best model")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")
    
    # Data processing parameters
    parser.add_argument("--feature_selection", action="store_true", default=True,
                        help="Use feature selection")
    parser.add_argument("--no_feature_selection", action="store_false", dest="feature_selection",
                        help="Disable feature selection")
    parser.add_argument("--k_features", type=int, default=50,
                        help="Number of features to select")
    parser.add_argument("--time_steps", type=int, default=10,
                        help="Number of time steps for LSTM sequences")
    parser.add_argument("--overlap", type=float, default=0.25,
                        help="Overlap fraction for sequence creation")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_split", type=float, default=0.6,
                        help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of epochs")
    
    return parser.parse_args()

# ---- Data Processing Functions ----
def preprocess(data, feature_selection=True, k_features=50):
    """Preprocess data: handle missing values, scale, select features, encode labels"""
    X = data[:, 2:].astype(np.float32)
    y = data[:, 0]
    
    # Handle missing values
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    
    # Scale features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Feature selection
    if feature_selection and X.shape[1] > k_features:
        selector = SelectKBest(f_classif, k=k_features)
        X = selector.fit_transform(X, y)
        print(f"Selected {X.shape[1]} features out of {data[:, 2:].shape[1]}")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

def reshape_with_overlap(X, y, time_steps=10, overlap=0.25):
    """Reshape data into overlapping sequences for LSTM"""
    step = int(time_steps * (1-overlap))
    segments, labels = [], []
    
    for i in range(0, len(X) - time_steps + 1, step):
        segments.append(X[i:i+time_steps])
        labels.append(np.bincount(y[i:i+time_steps]).argmax())  # Most frequent label
    
    return np.array(segments), np.array(labels)

# ---- Model Definition ----
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_dim//2)
        self.fc = nn.Linear(hidden_dim//2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Last time step
        out = self.batch_norm(out)
        return self.fc(out)

# ---- Training Function ----
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, args):
    """Train the model with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    train_accs, val_accs = [], []
    train_losses, val_losses = [], []
    
    print("Starting training...")
    for epoch in range(args.max_epochs):
        # Training phase
        model.train()
        total, correct, train_loss = 0, 0, 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
        train_acc = correct / total
        train_accs.append(train_acc)
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
    
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_total, val_correct, val_loss = 0, 0, 0
            for xb, yb in val_loader:
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.size(0)
            
            val_acc = val_correct / val_total
            val_accs.append(val_acc)
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1:2d}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.3f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.3f}, LR = {current_lr:.6f}")
    
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"✓ Model saved (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered!")
                break
    
    print("Training completed!")
    return train_accs, val_accs, train_losses, val_losses

# ---- Evaluation Function ----
def evaluate(model, test_loader, criterion, le, args):
    """Evaluate model on test set"""
    model.eval()
    all_preds, all_labels = [], []
    test_loss = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            test_loss += criterion(out, yb).item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(yb.numpy())
    
    test_loss = test_loss / len(test_loader)
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    print(f"\n✅ Final Test Results:")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Test Accuracy: {test_accuracy:.4f}")
    
    # Find classes present in test data
    unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
    valid_target_names = [str(le.classes_[i]) for i in unique_labels]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              labels=unique_labels, 
                              target_names=valid_target_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_target_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation=90)
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/confusion_matrix.png")
    plt.show()
    
    return all_preds, all_labels, test_accuracy

# ---- Main Execution ----
def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Print active configuration
    print("\n=== Configuration ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("====================\n")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    data = np.load(args.data_path, allow_pickle=True)
    print(f"Loaded data shape: {data.shape}")
    
    # Preprocess data
    X, y, le = preprocess(data, args.feature_selection, args.k_features)
    print(f"Classes: {le.classes_}")
    print(f"Feature shape after preprocessing: {X.shape}")
    
    # Create sequences
    X_seq, y_seq = reshape_with_overlap(X, y, args.time_steps, args.overlap)
    print(f"Sequence shape: {X_seq.shape}, Labels shape: {y_seq.shape}")
    
    # Create datasets
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y_seq, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split data
    total = len(dataset)
    train_size = int(args.train_split * total)
    val_size = int(args.val_split * total)
    test_size = total - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    # Initialize model
    input_dim = X_tensor.shape[2]
    model = LSTMClassifier(
        input_dim, 
        args.hidden_dim, 
        len(le.classes_), 
        args.dropout_rate
    )
    print(f"Model initialized with {input_dim} input features, {args.hidden_dim} hidden units, {len(le.classes_)} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train model
    train_accs, val_accs, train_losses, val_losses = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, args
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/training_curves.png")
    plt.show()
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(args.model_save_path))
    evaluate(model, test_loader, criterion, le, args)

if __name__ == '__main__':
    main()
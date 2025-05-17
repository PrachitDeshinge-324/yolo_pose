import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from utils.skeleton_gait import SkeletonSequence, SkeletonLSTM

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM model on skeleton sequences")
    parser.add_argument("--sequences", type=str, required=True, 
                       help="Path to saved skeleton sequences (.npy file)")
    parser.add_argument("--identities", type=str, default="person_identities.json",
                       help="Path to identities JSON file")
    parser.add_argument("--output", type=str, default="lstm_model",
                       help="Directory to save model and results")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="LSTM hidden state size")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--seq_length", type=int, default=30,
                       help="Fixed sequence length for LSTM input")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Proportion of data to use for testing")
    return parser.parse_args()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, output_dir):
    """Train the LSTM model"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track training progress
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for sequences, labels in train_loader:
            # Move data to device
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate accuracy
        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"Saved new best model with accuracy: {val_acc:.4f}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"))
    
    return model, history

def evaluate_model(model, test_loader, device, label_mapping, output_dir):
    """Evaluate the trained model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[label_mapping[i] for i in range(len(label_mapping))],
               yticklabels=[label_mapping[i] for i in range(len(label_mapping))])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    
    # Save test results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open(os.path.join(output_dir, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=== Training LSTM Model for Gait Recognition ===")
    
    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load sequences
    print("\n1. Loading skeleton sequences...")
    if not os.path.exists(args.sequences):
        print(f"Error: Sequences file {args.sequences} not found")
        return
    
    sequences = np.load(args.sequences, allow_pickle=True).item()
    print(f"Loaded {len(sequences)} sequence tracks")
    
    # Load identities
    print("\n2. Loading identity mappings...")
    if not os.path.exists(args.identities):
        print(f"Error: Identities file {args.identities} not found")
        return
    
    with open(args.identities, 'r') as f:
        identities = json.load(f)
        # Convert string keys to integers
        identities = {int(k): v for k, v in identities.items()}
    
    print(f"Loaded {len(identities)} identity mappings")
    
    # Create dataset
    print("\n3. Creating dataset...")
    dataset = SkeletonSequence(sequences, identities, seq_length=args.seq_length)
    
    # Split data
    print(f"\n4. Splitting dataset (test size: {args.test_size})...")
    test_size = int(len(dataset) * args.test_size)
    val_size = int((len(dataset) - test_size) * 0.1)  # 10% of train data for validation
    train_size = len(dataset) - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Get input size from first sequence
    first_seq, _ = dataset[0]
    input_size = first_seq.shape[-1]  # Last dimension is feature size
    
    # Initialize model
    print("\n5. Initializing LSTM model...")
    model = SkeletonLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=dataset.get_num_classes()
    ).to(device)
    
    print(f"Model architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    print(f"\n6. Training for {args.epochs} epochs...")
    model, history = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device,
        args.epochs, args.output
    )
    
    # Load best model for evaluation
    best_model_path = os.path.join(args.output, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate model
    print("\n7. Evaluating model on test set...")
    evaluate_model(model, test_loader, device, dataset.get_label_mapping(), args.output)
    
    print(f"\nModel training and evaluation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()
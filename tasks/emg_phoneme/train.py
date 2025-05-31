"""
Training script for EMG phoneme recognition using CTM
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from models.ctm_emg import EMGContinuousThoughtMachine
from tasks.emg_phoneme.modules import OnsetDetector, apply_cwt_to_segments, prepare_features_for_ctm
from data.custom_datasets import EMGCTMDataset
from tasks.emg_phoneme.utils import preprocess_emg_data

def emg_ctm_loss(predictions: torch.Tensor, 
                 certainties: torch.Tensor, 
                 targets: torch.Tensor) -> torch.Tensor:
    """
    CTM's dual-criterion loss for EMG phoneme prediction
    
    Args:
        predictions: (batch, num_classes, internal_ticks)
        certainties: (batch, 2, internal_ticks) - [entropy, 1-entropy]
        targets: (batch,) - phoneme class indices
    """
    B, C, T = predictions.shape
    
    # Expand targets across all time steps
    targets_exp = targets.unsqueeze(-1).expand(-1, T)
    
    # Compute cross-entropy loss at each internal tick
    predictions_flat = predictions.transpose(1, 2).contiguous().view(-1, C)
    targets_flat = targets_exp.contiguous().view(-1)
    
    losses_flat = F.cross_entropy(predictions_flat, targets_flat, reduction='none')
    losses = losses_flat.view(B, T)
    
    # Get certainty scores (1 - normalized entropy)
    certainty_scores = certainties[:, 1, :]
    
    # Find best time steps for each sample
    min_loss_idx = losses.argmin(dim=-1)
    max_cert_idx = certainty_scores.argmax(dim=-1)
    
    # Gather losses at optimal time steps
    batch_indices = torch.arange(B, device=predictions.device)
    min_losses = losses[batch_indices, min_loss_idx]
    max_cert_losses = losses[batch_indices, max_cert_idx]
    
    # CTM's dual criterion
    final_loss = (min_losses + max_cert_losses) / 2
    
    return final_loss.mean()

def compute_accuracy(predictions: torch.Tensor, 
                    certainties: torch.Tensor, 
                    labels: torch.Tensor) -> float:
    """Compute accuracy using CTM's certainty-based approach"""
    # Use maximum certainty predictions
    cert_scores = certainties[:, 1, :]
    max_cert_idx = cert_scores.argmax(dim=-1)
    batch_indices = torch.arange(len(labels), device=predictions.device)
    final_preds = predictions[batch_indices, :, max_cert_idx]
    pred_classes = final_preds.argmax(dim=1)
    accuracy = (pred_classes == labels).float().mean()
    
    return accuracy.item()

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for cwt_features, labels in tqdm(dataloader, desc="Training"):
        cwt_features = cwt_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions, certainties, _ = model(cwt_features)
        
        # Compute loss
        loss = emg_ctm_loss(predictions, certainties, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for CTM)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            accuracy = compute_accuracy(predictions, certainties, labels)
        
        running_loss += loss.item()
        running_accuracy += accuracy
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_accuracy / len(dataloader)
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set"""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    
    for cwt_features, labels in tqdm(dataloader, desc="Validation"):
        cwt_features = cwt_features.to(device)
        labels = labels.to(device)
        
        predictions, certainties, _ = model(cwt_features)
        
        loss = emg_ctm_loss(predictions, certainties, labels)
        accuracy = compute_accuracy(predictions, certainties, labels)
        
        running_loss += loss.item()
        running_accuracy += accuracy
    
    val_loss = running_loss / len(dataloader)
    val_acc = running_accuracy / len(dataloader)
    
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description='Train EMG-CTM for phoneme recognition')
    parser.add_argument('--data_path', type=str, required=True, help='Path to EMG dataset')
    parser.add_argument('--trials', nargs='+', default=['trial_3', 'trial_4'], help='Trial names to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_path', type=str, default='emg_ctm_model.pth', help='Model save path')
    
    args = parser.parse_args()
    
    # Load phoneme mapping
    phoneme_maps = {
        '[SIL]': 0, 'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5,
        'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11,
        'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17,
        'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
        'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29,
        'SH': 30, 'T': 31, 'TH': 32, 'UH': 33, 'UW': 34, 'V': 35,
        'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39
    }
    
    # Create dataset
    print("Loading dataset...")
    dataset = EMGCTMDataset(
        json_data_path=args.data_path,
        trial_names=args.trials,
        phoneme_maps=phoneme_maps
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing CTM...")
    model = EMGContinuousThoughtMachine(
        iterations=75,
        d_model=1024,
        d_input=512,
        heads=8,
        n_synch_out=512,
        n_synch_action=256,
        synapse_depth=8,
        memory_length=30,
        deep_nlms=True,
        memory_hidden_dims=64,
        out_dims=len(phoneme_maps),
        dropout=0.1,
        neuron_select_type='random-pairing',
        n_random_pairing_self=32
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, args.device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()

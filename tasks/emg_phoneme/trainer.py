"""Training utilities for EMG-CTM"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

class EMGCTMTrainer:
    """Trainer class for EMG-CTM experiments"""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler=None,
        device: str = 'cuda',
        grad_clip_norm: float = 20.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        
        self.model.to(device)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        
        for cwt_features, labels in tqdm(self.train_loader, desc="Training"):
            cwt_features = cwt_features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            loss, accuracy = self.train_step(cwt_features, labels)
            
            running_loss += loss
            running_accuracy += accuracy
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_accuracy / len(self.train_loader)
        
        return epoch_loss, epoch_acc
    

    def ctm_dual_criterion_loss(self, predictions, certainties, targets):
        """
        Exact CTM loss implementation from paper Section 2.5
        
        Args:
            predictions: (batch, num_classes, internal_ticks)
            certainties: (batch, 2, internal_ticks) - [entropy, 1-entropy]  
            targets: (batch,) - ground truth class indices
        """
        B, C, T = predictions.shape
        
        # Compute cross-entropy loss at each internal tick
        losses_per_tick = []
        for t in range(T):
            loss_t = F.cross_entropy(predictions[:, :, t], targets, reduction='none')
            losses_per_tick.append(loss_t)
        
        losses = torch.stack(losses_per_tick, dim=1)  # (batch, ticks)
        
        # CTM's dual criterion from paper
        min_loss_indices = losses.argmin(dim=1)  # (batch,)
        max_cert_indices = certainties[:, 1, :].argmax(dim=1)  # Use 1-entropy (batch,)
        
        # Gather losses at optimal points
        batch_indices = torch.arange(B, device=predictions.device)
        min_losses = losses[batch_indices, min_loss_indices]
        max_cert_losses = losses[batch_indices, max_cert_indices]
        
        # CTM paper: average of min loss and max certainty losses
        final_loss = (min_losses + max_cert_losses) / 2.0
        
        return final_loss.mean()

    def train_stepv2(self, model, optimizer, cwt_features, labels, device='cuda'):
        """
        CTM training step following paper specifications
        """
        self.model.train()
        optimizer.zero_grad()
        
        cwt_features = cwt_features.to(device)
        labels = labels.to(device)
        
        # Forward pass - CTM returns (predictions, certainties, sync_out)
        predictions, certainties, _ = model(cwt_features)
        
        # CTM's exact dual-criterion loss from paper
        loss = self.ctm_dual_criterion_loss(predictions, certainties, labels)
        
        # Backward pass
        loss.backward()
        
        # CRITICAL: CTM paper uses gradient norm clipping = 20
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        
        # Check for gradient health (important for CTM stability)
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        optimizer.step()
        
        return loss.item(), total_grad_norm


    def train_step(self, cwt_features: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Forward pass through CTM
        predictions, certainties, _ = self.model(cwt_features)
        
        # CTM's dual-criterion loss
        loss = self.ctm_loss(predictions, certainties, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for CTM stability)
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        
        self.optimizer.step()
        
        # Compute accuracy using CTM's approach
        with torch.no_grad():
            accuracy = self.compute_accuracy(predictions, certainties, labels)
        
        return loss.item(), accuracy
    
    def ctm_loss(
        self, 
        predictions: torch.Tensor, 
        certainties: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        CTM's dual-criterion loss function
        
        Args:
            predictions: (batch, num_classes, internal_ticks)
            certainties: (batch, 2, internal_ticks) - [entropy, 1-entropy]
            targets: (batch,) - ground truth labels
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
    
    def compute_accuracy(
        self, 
        predictions: torch.Tensor, 
        certainties: torch.Tensor, 
        labels: torch.Tensor
    ) -> float:
        """Compute accuracy using CTM's certainty-based approach"""
        # Use maximum certainty predictions
        cert_scores = certainties[:, 1, :]
        max_cert_idx = cert_scores.argmax(dim=-1)
        batch_indices = torch.arange(len(labels), device=predictions.device)
        final_preds = predictions[batch_indices, :, max_cert_idx]
        pred_classes = final_preds.argmax(dim=1)
        accuracy = (pred_classes == labels).float().mean()
        
        return accuracy.item()
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        running_loss = 0.0
        running_accuracy = 0.0
        
        for cwt_features, labels in tqdm(self.val_loader, desc="Validation"):
            cwt_features = cwt_features.to(self.device)
            labels = labels.to(self.device)
            
            predictions, certainties, _ = self.model(cwt_features)
            
            loss = self.ctm_loss(predictions, certainties, labels)
            accuracy = self.compute_accuracy(predictions, certainties, labels)
            
            running_loss += loss.item()
            running_accuracy += accuracy
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = running_accuracy / len(self.val_loader)
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int) -> Dict:
        """Complete training loop"""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.evaluate()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_emg_ctm.pth')
                print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        return history

def ctm_emg_loss(predictions, targets, certainties):
    """
    CTM's dual-criterion loss for EMG phoneme recognition
    
    Args:
        predictions: (batch, num_classes=38, internal_ticks)
        targets: (batch,) - phoneme class indices
        certainties: (batch, 2, internal_ticks) - [entropy, 1-entropy]
    """
    B, C, T = predictions.shape
    
    # Expand targets across internal ticks
    targets_expanded = targets.unsqueeze(-1).expand(-1, T)
    
    # Compute loss at each internal tick
    losses = F.cross_entropy(predictions, targets_expanded, reduction='none')  # (B, T)
    
    # CTM's dual criterion - this is CRITICAL
    min_loss_indices = losses.argmin(dim=-1)  # (B,)
    max_cert_indices = certainties[:, 1, :].argmax(dim=-1)  # (B,) - use 1-entropy
    
    # Gather losses at critical points
    batch_idx = torch.arange(B, device=predictions.device)
    min_losses = losses[batch_idx, min_loss_indices]
    max_cert_losses = losses[batch_idx, max_cert_indices]
    
    # CTM's aggregation strategy
    final_loss = (min_losses + max_cert_losses) / 2
    
    return final_loss.mean(), {
        'min_loss_tick': min_loss_indices.float().mean(),
        'max_cert_tick': max_cert_indices.float().mean(),
        'avg_certainty': certainties[:, 1, :].mean()
    }

def train_emg_ctm(
    model, 
    train_dataset, 
    val_dataset,
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    warmup_steps=10000,
    grad_clip_norm=20.0,  # CTM paper uses 20
    device='cuda'
):
    """
    Complete training loop following CTM best practices
    """
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer (CTM paper uses AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0001  # Light regularization for EMG
    )
    
    # Learning rate scheduler (CTM uses cosine annealing with warmup)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy='cos'
    )
    
    model.to(device)
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_metrics = {
            'min_loss_tick': 0.0,
            'max_cert_tick': 0.0,
            'avg_certainty': 0.0,
            'grad_norm': 0.0
        }
        
        for batch_idx, (cwt_features, labels) in enumerate(train_loader):
            cwt_features = cwt_features.to(device)
            labels = labels.to(device)
            
            # Training step
            loss, metrics, grad_norm = train_emg_ctm_step(
                model, optimizer, cwt_features, labels, grad_clip_norm
            )
            
            scheduler.step()
            
            # Accumulate metrics
            train_loss += loss
            for key in train_metrics:
                if key == 'grad_norm':
                    train_metrics[key] += grad_norm
                else:
                    train_metrics[key] += metrics[key.replace('_', '_')]
            
            # Log progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}, '
                      f'Grad Norm: {grad_norm:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        val_acc, val_loss = evaluate_emg_ctm(model, val_loader, device)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Avg Certainty: {train_metrics["avg_certainty"]:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emg_ctm.pth')
            print(f'New best validation accuracy: {best_val_acc:.4f}')

def evaluate_emg_ctm(model, dataloader, device):
    """Evaluation function for CTM"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for cwt_features, labels in dataloader:
            cwt_features = cwt_features.to(device)
            labels = labels.to(device)
            
            predictions, certainties, _ = model(cwt_features)
            
            # Use most certain prediction for evaluation
            most_certain_tick = certainties[:, 1, :].argmax(dim=-1)
            batch_idx = torch.arange(predictions.size(0), device=device)
            final_predictions = predictions[batch_idx, :, most_certain_tick]
            
            loss, _ = ctm_emg_loss(predictions, labels, certainties)
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = final_predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss

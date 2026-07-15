import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


class LightMedSegLoss(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.num_classes = num_classes
        
    def forward(self, logits, targets):
        B, C, D, H, W = logits.shape
        
        # Small value to avoid division by zero
        eps = 2e-5
        
        # Per-class weights calculation:
        # ______________________________
        # Total number of voxels across batches
        omega = B * D * H * W
        # Number of target voxels per class 
        counts = targets.sum(dim=(0, 2, 3, 4))
        # weights per class (biased by number of relevant voxels per class)
        w_c = omega / (self.num_classes * counts + eps)
        # normalize weights
        w_c = w_c / w_c.sum() * self.num_classes
        
        # Cross entropy loss (weighted)
        loss_ce = F.cross_entropy(logits, targets, weight=w_c)
        
        # DICE loss calculation:
        # ______________________________
        # Calculate class probabilities per voxel with softmax
        probs = F.softmax(logits, dim=1)
        # Calculate the intersection between predicted masks and target masks per class
        intersection = torch.sum(probs * targets, dim=(2, 3, 4))
        # Calculate the total area of predicted and target masks per class
        total = torch.sum(probs, dim=(2, 3, 4)) + torch.sum(targets, dim=(2, 3, 4))
        # Calculate the DICE-score
        dice_score = (2. * intersection + eps) / (total + eps)
        # Subtract cross-batch mean from 1.0 for dice loss per class
        dice_loss_per_class = 1.0 - dice_score.mean(dim=0)
        # Apply per-class weights to calculate final dice loss
        loss_dice = torch.sum(dice_loss_per_class * w_c) / self.num_classes
        
        
        # Boundary loss:
        # ______________________________
        # Boundary mask
        dilated = F.max_pool3d(targets, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool3d(-targets, kernel_size=3, stride=1, padding=1)
        M = dilated - eroded
        # Independently calculates the cross-entropy for each class
        bce_voxelwise = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # Masks out the boundary voxels
        masked_bce = bce_voxelwise * M
        # Total boundary voxels per batch & class
        M_sum = torch.sum(M, dim=(2, 3, 4)) + eps
        # Normalized cross entropy per batch and class
        bdry_loss_per_item_class = torch.sum(masked_bce, dim=(2, 3, 4)) / M_sum
        # Normalized cross entropy per class
        bdry_loss_per_class = bdry_loss_per_item_class.mean(dim=0)
        # Boundary loss (weighted)
        loss_bdry = torch.sum(bdry_loss_per_class * w_c) / self.num_classes
        
        # Total loss calculation
        total_loss = loss_dice + loss_ce + 0.5 * loss_bdry
        
        return total_loss, loss_dice, loss_ce, loss_bdry
        

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs=100,
    # learning rate range initial (max) to final (min)
    lr=(2e-4, 1e-9),
    device="cuda",
    save_path_best="lightmedseg_best.pth",
    save_path_last="lightmedseg_last.pth"
):
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr[0], weight_decay=1e-5)
    criterion = LightMedSegLoss(num_classes=2) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=lr[1], T_max=num_epochs)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n")
        
        # ==============
        # Training phase
        # ==============
        model.train()
        
        train_loss, train_dice, train_ce, train_bdry = 0.0, 0.0, 0.0, 0.0
        
        train_loop = tqdm(train_loader, desc='Train')
        
        for batch in train_loop:
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            targets = batch['mask'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device, dtype=torch.float32):
                logits = model(images, metadata)
                loss, l_dice, l_ce, l_bdry = criterion(logits, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_dice += l_dice.item()
            train_ce += l_ce.item()
            train_bdry += l_bdry.item()
            
            free_mem, total_mem = torch.cuda.mem_get_info()
            
            used_mem = (total_mem - free_mem) / 2 ** 20
            total_mem = total_mem / 2 ** 20
            
            train_loop.set_postfix(
                Tot=f"{loss.item():.3f}", 
                Dice=f"{l_dice.item():.3f}", 
                CE=f"{l_ce.item():.3f}", 
                Bdry=f"{l_bdry.item():.3f}",
                Mem=f"Used VRAM: {used_mem}MiB/{total_mem}MiB"
            )
            
        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches
        scheduler.step()
        
        # ================
        # Validation phase
        # ================
        
        model.eval()
        val_loss, val_dice, val_ce, val_bdry = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc = 'Val')
            
            for batch in val_loop:
                images = batch['image'].to(device)
                metadata = batch['metadata'].to(device)
                targets=  batch['mask'].to(device)
                
                with autocast(device_type=device, dtype=torch.float32):
                    logits = model(images, metadata)
                    loss, l_dice, l_ce, l_bdry = criterion(logits, targets)
            
            val_loss += loss.item()
            val_dice += l_dice.item()
            val_ce += l_ce.item()
            val_bdry += l_bdry.item()
            
            val_loop.set_postfix(
                Tot=f"{loss.item():.3f}", 
                Dice=f"{l_dice.item():.3f}", 
                CE=f"{l_ce.item():.3f}", 
                Bdry=f"{l_bdry.item():.3f}"
            )
        
        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches
        current_lr = scheduler.get_last_lr()[0]
        
        print(
            f"Train | Tot: {avg_train_loss:.4f}  Dice: {train_dice/num_train_batches:.4f}  "
            f"CE: {train_ce/num_train_batches:.4f}  Bdry: {train_bdry/num_train_batches:.4f}"
        )
        print(
            f"Val   | Tot: {avg_val_loss:.4f}  Dice: {val_dice/num_val_batches:.4f}  "
            f"CE: {val_ce/num_val_batches:.4f}  Bdry: {val_bdry/num_val_batches:.4f} | LR: {current_lr:.2e}"
        )
        
        # ==========================
        #       CHECKPOINTING
        # ==========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Validation loss improved to {best_val_loss:.4f}. Saving checkpoint!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path_best)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path_last)

    
        
            
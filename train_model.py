import os
import torch
import argparse
from models.models import LightMedSeg, LMSBR
from utils.dataset import ISLESDataset, OctantCropDataset
from utils.loss import LightMedSegLoss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# from test_model import visualize_prediction

from tqdm import tqdm


def train_model(
    model: LightMedSeg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type,
    num_epochs=100,
    # learning rate range initial (max) to final (min)
    lr=(2e-4, 1e-9),
    ce_only=False,
    device="cuda",
    save_path_best="lightmedseg_best.pth",
    save_path_last="lightmedseg_last.pth"
):
    
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr[0], weight_decay=0)
    criterion = LightMedSegLoss(num_classes=2, ce_only=ce_only) 
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
                        
            # with autocast(device_type=device, dtype=torch.float32):
            logits = model(images, metadata)
            loss, l_dice, l_ce, l_bdry = criterion(logits, targets)
                
                
            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
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
                Mem=f"{used_mem}MiB/{total_mem}MiB"
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
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'model_type': model_type,
            'hyperparams': model.hyperparams()
        }   

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Validation loss improved to {best_val_loss:.4f}. Saving checkpoint!")
            torch.save(save_dict, save_path_best)

        torch.save(save_dict, save_path_last)
            


def main():
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        help="Where to output the best and last weights.",
        default="out",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs for training/eval.",
        type=int,
        default=40,
    )
    parser.add_argument("-b", "--batch-size", help="Batch size.", type=int, default=1)

    parser.add_argument(
        "-r",
        "--range",
        help="Range of datapoints to train on in the format start_idx:end_idx.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-a",
        "--num-anchors",
        help="num_anchors hyperparameter value.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Whether to use the base model or the one with boundary refinement.",
        default="base",
        choices=["base", "refined"],
    )
    parser.add_argument(
        "-d",
        "--ignore-metadata",
        help="Disables the metadata FiLM functionality.",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--crop", help="Train using random crop.", action="store_true"
    )

    args = parser.parse_args()

    output_dir = args.output
    epochs = args.epochs
    batch_size = args.batch_size
    num_anchors = args.num_anchors
    crop = args.crop
    add_edges = args.model == "refined"
    metadata_film = not args.ignore_metadata
    downsample = not crop

    rng = args.range
    data_range = None if rng == None else [int(idx) for idx in rng.split(":")]
    print(data_range)

    dataset = ISLESDataset(range=data_range, add_edges=add_edges, random_crop=True)

    print(len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # print(len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if args.model == "base":
        model = LightMedSeg(
            n_classes=2,
            in_channels=1,
            num_anchors=num_anchors,
            metadata_film=metadata_film,
            downsample=downsample,
        )
    else:
        model = LMSBR(
            n_classes=2,
            num_anchors=num_anchors,
            metadata_film=metadata_film,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_model(
        model,
        train_loader,
        val_loader,
        model_type=args.model,
        num_epochs=epochs,
        device=device,
        lr=(1e-3, 1e-8),
        # ce_only=True,
        save_path_best=os.path.join(output_dir, "best.pth"),
        save_path_last=os.path.join(output_dir, "last.pth"),
    )

    # image = dataset[0]["image"].to(device)
    # mask = dataset[0]["mask"].to(device)

    # visualize_prediction(model, image, mask)


if __name__ == "__main__":
    main()

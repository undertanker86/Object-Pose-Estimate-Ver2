import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import tomli
from tqdm import tqdm
import torch.nn as nn
from model import MTGOE
from dataset.bop_dataset import BOPDataset
from TeacherRenderer import TeacherRenderer
from self_supervision import TeacherStudentTrainer, GeometryGuidedFilter
from torch.cuda.amp import autocast, GradScaler

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_pos=1.0, lambda_depth=0.5, lambda_seg=0.5, lambda_vf=0.5):
        super(MultiTaskLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_depth = lambda_depth
        self.lambda_seg = lambda_seg
        self.lambda_vf = lambda_vf
        
    def forward(self, outputs, targets):
        """
        Compute multi-task loss
        
        Args:
            outputs: tuple of (vote_map, centers, translations, depth, seg, vector_field, fused_features)
            targets: dictionary with 'depth_map', 'segmentation_mask', 'vector_field', etc.
            
        Returns:
            total_loss: combined loss value
        """
        vote_map, centers, translations, depth, seg, vector_field, _ = outputs
        
        # Print debug info about predictions
        print(f"Debug - Predicted centers: {len(centers)}, translations: {len(translations)}")
        if len(centers) > 0:
            print(f"Debug - First center: {centers[0]}")
        if len(translations) > 0:
            print(f"Debug - First translation: {translations[0]}")
        
        # Initialize loss components with zeros
        pos_loss = torch.tensor(0.0, device=depth.device)
        center_loss = torch.tensor(0.0, device=depth.device)
        depth_loss = torch.tensor(0.0, device=depth.device)
        seg_loss = torch.tensor(0.0, device=depth.device)
        vf_loss = torch.tensor(0.0, device=depth.device)
        
        # Pose and center loss
        if 'pose' in targets and targets['pose'] is not None:
            # Check for gt_translations and centers
            gt_translations = targets['pose'].get('t')
            gt_centers = targets['pose'].get('centers')
            
            # Debug info about ground truth
            print(f"Debug - GT centers: {gt_centers.shape if gt_centers is not None else None}")
            print(f"Debug - GT translations: {gt_translations.shape if gt_translations is not None else None}")
            if gt_centers is not None and gt_centers.shape[0] > 0:
                print(f"Debug - First GT center: {gt_centers[0]}")
            if gt_translations is not None and gt_translations.shape[0] > 0:
                print(f"Debug - First GT translation: {gt_translations[0]}")
            
            # Position loss
            if gt_translations is not None and len(translations) > 0:
                if gt_translations.shape[0] > 0:  # Ensure we have at least one ground truth translation
                    pos_errors = []
                    for pred_t in translations:
                        # Find closest ground truth
                        dists = torch.norm(pred_t.unsqueeze(0) - gt_translations, dim=1)
                        if dists.numel() > 0:  # Check if dists is not empty
                            min_dist = torch.min(dists)
                            pos_errors.append(min_dist)
                    
                    if pos_errors:
                        pos_loss = torch.stack(pos_errors).mean()
                        print(f"Debug - Position loss: {pos_loss.item():.6f}")
            
            # Center loss
            if gt_centers is not None and len(centers) > 0:
                if gt_centers.shape[0] > 0:  # Ensure we have at least one ground truth center
                    center_errors = []
                    for pred_c in centers:
                        # Find closest ground truth
                        dists_c = torch.norm(pred_c.unsqueeze(0) - gt_centers, dim=1)
                        if dists_c.numel() > 0:  # Check if dists_c is not empty
                            min_dist_c = torch.min(dists_c)
                            center_errors.append(min_dist_c)
                    
                    if center_errors:
                        center_loss = torch.stack(center_errors).mean()
                        print(f"Debug - Center loss: {center_loss.item():.6f}")
        
        # Depth loss
        if 'depth_map' in targets and targets['depth_map'] is not None:
            # Ensure dimensions match
            target_depth = targets['depth_map']
            if target_depth.dim() == 2:  # Add batch and channel dimensions if needed
                target_depth = target_depth.unsqueeze(0).unsqueeze(0)
            elif target_depth.dim() == 3:  # Add channel dimension if needed
                target_depth = target_depth.unsqueeze(1)
            
            # Interpolate if sizes don't match
            if target_depth.shape[2:] != depth.shape[2:]:
                target_depth = F.interpolate(
                    target_depth, 
                    size=depth.shape[2:], 
                    mode='nearest'
                )
            
            # Normalize target_depth if it's in different scale than predicted depth
            if target_depth.max() > 10.0 and depth.max() < 10.0:  # Check if scales are different
                target_depth = target_depth / 255.0  # Normalize to 0-1 range if needed
            
            depth_loss = F.l1_loss(depth, target_depth)
            print(f"Debug - Depth loss: {depth_loss.item():.6f}, depth range: [{depth.min().item():.4f}, {depth.max().item():.4f}], target range: [{target_depth.min().item():.4f}, {target_depth.max().item():.4f}]")
        
        # Segmentation loss
        if 'segmentation_mask' in targets and targets['segmentation_mask'] is not None:
            target_seg = targets['segmentation_mask']
            
            # Ensure target is proper shape for cross entropy
            if target_seg.dim() == 4 and target_seg.shape[1] == 1:
                target_seg = target_seg.squeeze(1)
            
            # Interpolate if sizes don't match
            if target_seg.shape[-2:] != seg.shape[-2:]:
                target_seg = F.interpolate(
                    target_seg.unsqueeze(1).float(), 
                    size=seg.shape[-2:], 
                    mode='nearest'
                ).squeeze(1).long()
            
            seg_loss = F.cross_entropy(seg, target_seg.long())
            print(f"Debug - Segmentation loss: {seg_loss.item():.6f}")
        
        # Vector field loss
        if 'vector_field' in targets and targets['vector_field'] is not None:
            gt_vector_field = targets['vector_field']
            
            # Reshape ground truth vector field if needed
            if gt_vector_field.dim() == 4 and gt_vector_field.shape[1] == 2:
                # Already in correct format [B, 2, H, W]
                pass
            elif gt_vector_field.dim() == 4 and gt_vector_field.shape[-1] == 2:
                # Format [B, H, W, 2] -> [B, 2, H, W]
                gt_vector_field = gt_vector_field.permute(0, 3, 1, 2)
            
            # Create mask for valid vector field pixels
            if 'segmentation_mask' in targets and targets['segmentation_mask'] is not None:
                mask = targets['segmentation_mask'].float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
            else:
                mask = torch.ones_like(vector_field[:, :1])
            
            # Ensure mask and vector field have same spatial dimensions
            if mask.shape[2:] != vector_field.shape[2:]:
                mask = F.interpolate(mask, size=vector_field.shape[2:], mode='nearest')
            
            # Ensure ground truth vector field has same spatial dimensions
            if gt_vector_field.shape[2:] != vector_field.shape[2:]:
                gt_vector_field = F.interpolate(
                    gt_vector_field, 
                    size=vector_field.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate masked loss
            valid_pixels = mask.sum() + 1e-10  # Avoid division by zero
            vf_loss = F.smooth_l1_loss(
                vector_field * mask, 
                gt_vector_field * mask, 
                reduction='sum'
            ) / valid_pixels
            print(f"Debug - Vector field loss: {vf_loss.item():.6f}")
        
        # Combine losses with weights
        total_loss = (
            self.lambda_pos * pos_loss +
            self.lambda_depth * depth_loss +
            self.lambda_seg * seg_loss +
            self.lambda_vf * vf_loss +
            0.5 * center_loss  # Weight for center loss
        )
        
        # Log individual loss components
        loss_dict = {
            'pos': pos_loss.item(),
            'depth': depth_loss.item(),
            'seg': seg_loss.item(),
            'vf': vf_loss.item(),
            'center': center_loss.item(),
            'total': total_loss.item()
        }
        
        # Print summary
        print(f"Loss components: pos={pos_loss.item():.4f}, depth={depth_loss.item():.4f}, "
              f"seg={seg_loss.item():.4f}, vf={vf_loss.item():.4f}, center={center_loss.item():.4f}")
        
        return total_loss, loss_dict

def debug_loss_components(model, batch, multi_task_loss):
    """
    Function to debug each component of the loss function individually
    
    Args:
        model: The neural network model
        batch: Batch of data
        multi_task_loss: Loss function
        
    Returns:
        None, prints debug information
    """
    # Ensure batch is on the correct device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Extract camera parameters
    camera_params = {}
    if 'cam_K' in batch:
        camera_params = {
            'fx': batch['cam_K'][:, 0, 0],
            'fy': batch['cam_K'][:, 1, 1],
            'px': batch['cam_K'][:, 0, 2],
            'py': batch['cam_K'][:, 1, 2]
        }
    
    # Check segmentation map and vectors before model forward
    if 'segmentation_mask' in batch and batch['segmentation_mask'] is not None:
        seg_map = batch['segmentation_mask']
        print(f"Segmentation map: sum={seg_map.sum().item()}, max={seg_map.max().item()}")
    
    if 'vector_field' in batch and batch['vector_field'] is not None:
        vf = batch['vector_field']
        if vf.dim() == 4 and vf.shape[-1] == 2:  # [B, H, W, 2] format
            print(f"Direction vectors: mean magnitude={torch.norm(vf, dim=3).mean().item()}")
        elif vf.dim() == 4 and vf.shape[1] == 2:  # [B, 2, H, W] format
            print(f"Direction vectors: mean magnitude={torch.norm(vf, dim=1).mean().item()}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch['rgb'], **camera_params)
    
    # Unpack outputs
    vote_map, centers, translations, depth, seg, vector_field, features = outputs
    
    print("\n===== MODEL OUTPUTS =====")
    print(f"vote_map: shape={vote_map.shape}, range=[{vote_map.min().item():.4f}, {vote_map.max().item():.4f}]")
    print(f"centers: {len(centers)} centers detected")
    if len(centers) > 0:
        print(f"  First center: {centers[0]}")
    print(f"translations: {len(translations)} translations detected")
    if len(translations) > 0:
        print(f"  First translation: {translations[0]}")
    print(f"depth: shape={depth.shape}, range=[{depth.min().item():.4f}, {depth.max().item():.4f}]")
    print(f"seg: shape={seg.shape}, range=[{seg.min().item():.4f}, {seg.max().item():.4f}]")
    print(f"vector_field: shape={vector_field.shape}, range=[{vector_field.min().item():.4f}, {vector_field.max().item():.4f}]")
    
    # Create targets
    targets = {}
    
    # Depth map
    if 'depth_map' in batch and batch['depth_map'] is not None:
        depth_map = batch['depth_map']
        if depth_map.dim() == 2:
            depth_map = depth_map.unsqueeze(0)
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        targets['depth_map'] = depth_map
        print(f"\ndepth_map: shape={depth_map.shape}, range=[{depth_map.min().item():.4f}, {depth_map.max().item():.4f}]")
        
        # Test depth loss independently
        if depth_map.shape[2:] != depth.shape[2:]:
            depth_map = F.interpolate(depth_map, size=depth.shape[2:], mode='nearest')
        depth_loss = F.l1_loss(depth, depth_map)
        print(f"Depth loss (independent): {depth_loss.item():.6f}")
    
    # Segmentation mask
    if 'segmentation_mask' in batch and batch['segmentation_mask'] is not None:
        seg_mask = batch['segmentation_mask']
        targets['segmentation_mask'] = seg_mask
        print(f"\nsegmentation_mask: shape={seg_mask.shape}, unique_values={torch.unique(seg_mask).tolist()}")
        
        # Test segmentation loss independently
        if seg_mask.dim() == 4 and seg_mask.shape[1] == 1:
            seg_mask = seg_mask.squeeze(1)
        if seg_mask.shape[-2:] != seg.shape[-2:]:
            seg_mask = F.interpolate(seg_mask.unsqueeze(1).float(), size=seg.shape[-2:], mode='nearest').squeeze(1).long()
        try:
            seg_loss = F.cross_entropy(seg, seg_mask.long())
            print(f"Segmentation loss (independent): {seg_loss.item():.6f}")
        except Exception as e:
            print(f"Error computing segmentation loss: {e}")
            print(f"  seg shape: {seg.shape}, seg_mask shape: {seg_mask.shape}")
    
    # Vector field
    if 'vector_field' in batch and batch['vector_field'] is not None:
        vf = batch['vector_field']
        targets['vector_field'] = vf
        print(f"\nvector_field: shape={vf.shape}, range=[{vf.min().item():.4f}, {vf.max().item():.4f}]")
        
        # Test vector field loss independently
        if vf.dim() == 4 and vf.shape[-1] == 2:
            vf = vf.permute(0, 3, 1, 2)
        if vf.shape[2:] != vector_field.shape[2:]:
            vf = F.interpolate(vf, size=vector_field.shape[2:], mode='bilinear', align_corners=False)
        
        # Use segmentation mask as weight
        if 'segmentation_mask' in batch and batch['segmentation_mask'] is not None:
            mask = batch['segmentation_mask'].float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[2:] != vector_field.shape[2:]:
                mask = F.interpolate(mask, size=vector_field.shape[2:], mode='nearest')
        else:
            mask = torch.ones_like(vector_field[:, :1])
        
        try:
            vf_loss = F.smooth_l1_loss(vector_field * mask, vf * mask, reduction='sum') / (mask.sum() + 1e-10)
            print(f"Vector field loss (independent): {vf_loss.item():.6f}")
        except Exception as e:
            print(f"Error computing vector field loss: {e}")
            print(f"  vector_field shape: {vector_field.shape}, vf shape: {vf.shape}, mask shape: {mask.shape}")
    
    # Pose
    pose_data = {}
    if 'cam_R_m2c' in batch and 'cam_t_m2c' in batch:
        pose_data['R'] = batch['cam_R_m2c']
        pose_data['t'] = batch['cam_t_m2c']
        print(f"\nPose data:")
        print(f"  R shape: {pose_data['R'].shape}")
        print(f"  t shape: {pose_data['t'].shape}")
    
    # 2D center points
    if 'gt_center_x' in batch and 'gt_center_y' in batch:
        gt_centers = torch.stack([batch['gt_center_x'], batch['gt_center_y']], dim=1)
        pose_data['centers'] = gt_centers
        print(f"  centers shape: {gt_centers.shape}")
        
        # Test center loss independently
        if len(centers) > 0 and len(gt_centers) > 0:
            center_errors = []
            for pred_c in centers:
                dists_c = torch.norm(pred_c.unsqueeze(0) - gt_centers, dim=1)
                min_dist_c = torch.min(dists_c)
                center_errors.append(min_dist_c)
            
            if center_errors:
                center_loss = torch.stack(center_errors).mean()
                print(f"Center loss (independent): {center_loss.item():.6f}")
    
    if pose_data:
        targets['pose'] = pose_data
    
    # Now run the complete loss function
    try:
        total_loss, loss_dict = multi_task_loss(outputs, targets)
        print(f"\nCombined loss: {total_loss.item():.6f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.6f}")
    except Exception as e:
        print(f"Error computing combined loss: {e}")
        import traceback
        traceback.print_exc()
# Create argument parser
parser = argparse.ArgumentParser(description='Train MTGOE model for 6D pose estimation')
parser.add_argument('--config', type=str, default='config.toml', help='Path to config file')
parser.add_argument('--obj_id', type=int, default=1, help='Object ID to train on')
parser.add_argument('--mode', type=str, choices=['supervised', 'self_supervised', 'both'], default='both',
                   help='Training mode: supervised, self_supervised, or both')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory for checkpoints')
args = parser.parse_args()

# Load config
with open(args.config, "rb") as f:
    config = tomli.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize datasets
# Paths adjusted for your dataset structure
# Đường dẫn tới dataset
synthetic_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='train',
    obj_id=1
)

real_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='train',
    obj_id=1
)

test_dataset = BOPDataset(
    dataset_root='./data/lmo',
    split='test',
    obj_id=1
)

# Create data loaders
synthetic_loader = DataLoader(
    synthetic_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

real_loader = DataLoader(
    real_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# Initialize models
mtgoe_model = MTGOE(
    image_height=config['data']['image_height'],
    image_width=config['data']['image_width']
).to(device)

# pose_model = PoseEstimationModel(
#     depth_channels=1,
#     seg_channels=19,
#     vector_channels=2,
#     feature_dim=256,
#     num_keypoints=8
# ).to(device)

# Initialize loss function
multi_task_loss = MultiTaskLoss(
    lambda_pos=config['lambdas']['lambda_pos'],
    lambda_depth=config['lambdas']['lambda_depth'],
    lambda_seg=config['lambdas']['lambda_sem'],
    lambda_vf=config['lambdas']['lambda_vf']
)

# Initialize optimizer
optimizer = optim.Adam(
    list(mtgoe_model.parameters()),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# Initialize learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['training']['scheduler_step_size'],
    gamma=config['training']['scheduler_gamma']
)

# Resume from checkpoint if specified
start_epoch = 0
if args.resume:
    if os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume)
        mtgoe_model.load_state_dict(checkpoint['mtgoe_state_dict'])
        # pose_model.load_state_dict(checkpoint['pose_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    else:
        print(f"No checkpoint found at {args.resume}")

# Function to save checkpoint
def save_checkpoint(epoch, is_best=False):
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'mtgoe_state_dict': mtgoe_model.state_dict(),
        # 'pose_state_dict': pose_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(args.output_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'mtgoe_state_dict': mtgoe_model.state_dict(),
            # 'pose_state_dict': pose_model.state_dict(),
        }, best_path)
    
    print(f"Checkpoint saved to {checkpoint_path}")

def train_supervised(num_epochs):
    """Train with synthetic data"""
    print(f"\nStarting supervised training for {num_epochs} epochs...")
    
    # Get a sample batch for debugging
    sample_batch = next(iter(synthetic_loader))
    print("\nTesting loss components before training...")
    debug_loss_components(mtgoe_model, sample_batch, multi_task_loss)
    
    mtgoe_model.train()
    scaler = GradScaler()  # Helps prevent underflow
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses = {'total': 0, 'pos': 0, 'depth': 0, 'seg': 0, 'vf': 0, 'center': 0}
        valid_batches = 0
        
        progress_bar = tqdm(enumerate(synthetic_loader), total=len(synthetic_loader), desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}")
        for batch_idx, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Skip incomplete batches
            if 'rgb' not in batch or batch['rgb'].shape[0] == 0:
                print("Skipping batch with missing or empty RGB data")
                continue
            
            # Extract camera parameters
            camera_params = {}
            if 'cam_K' in batch:
                camera_params = {
                    'fx': batch['cam_K'][:, 0, 0],
                    'fy': batch['cam_K'][:, 1, 1],
                    'px': batch['cam_K'][:, 0, 2],
                    'py': batch['cam_K'][:, 1, 2]
                }
            
            # Forward pass
            outputs = mtgoe_model(batch['rgb'], **camera_params)
            
            # Create targets with proper validation
            targets = {}
            
            # Depth map
            if 'depth_map' in batch and batch['depth_map'] is not None:
                targets['depth_map'] = batch['depth_map']
            
            # Segmentation mask
            if 'segmentation_mask' in batch and batch['segmentation_mask'] is not None:
                targets['segmentation_mask'] = batch['segmentation_mask']
            elif 'mask' in batch and batch['mask'] is not None:
                targets['segmentation_mask'] = batch['mask']  # Use regular mask if segmentation_mask not available
            
            # Vector field
            if 'vector_field' in batch and batch['vector_field'] is not None:
                targets['vector_field'] = batch['vector_field']
            
            # Pose information
            pose_data = {}
            if 'cam_R_m2c' in batch and 'cam_t_m2c' in batch:
                pose_data['R'] = batch['cam_R_m2c']
                pose_data['t'] = batch['cam_t_m2c']
            
            # 2D center points
            if 'gt_center_x' in batch and 'gt_center_y' in batch:
                centers = torch.stack([batch['gt_center_x'], batch['gt_center_y']], dim=1)
                pose_data['centers'] = centers
            
            if pose_data:
                targets['pose'] = pose_data
            
            # Compute loss
            loss, losses_dict = multi_task_loss(outputs, targets)
            



                
            # Skip if loss is invalid
            if not torch.isfinite(loss):
                print(f"Warning: Skipping batch {batch_idx} due to invalid loss value: {loss.item()}")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            # loss.backward()
            scaler.scale(loss).backward()  # Scale loss to prevent underflow
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(mtgoe_model.parameters()), 
                config['training']['gradient_clip']
            )
            
            # optimizer.step()
                      
            scaler.step(optimizer)  # Step optimizer
            scaler.update()  # Update scaler for next iteration
            
            # Update progress bar
            valid_batches += 1
            for k, v in losses_dict.items():
                if k in epoch_losses:
                    epoch_losses[k] += v
                
            # Calculate average loss from valid batches only
            avg_losses = {k: v / valid_batches for k, v in epoch_losses.items()}
            progress_bar.set_postfix({'loss': f"{avg_losses['total']:.4f}"})
        
        # Step the scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs} Results:")
        for k, v in epoch_losses.items():
            avg_v = v / max(1, valid_batches)
            print(f"  {k}: {avg_v:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            save_checkpoint(epoch)
    
    return start_epoch + num_epochs

# Function for self-supervised training
def train_self_supervised(num_epochs=20, start_from_epoch=0):
    """Self-supervised training with real data"""
    print(f"\nStarting self-supervised training for {num_epochs} epochs...")
    
    # Initialize teacher-student framework
    teacher_model = MTGOE(
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width']
    ).to(device)
    teacher_model.load_state_dict(mtgoe_model.state_dict())
    
    renderer = TeacherRenderer(
        cad_model_path=real_dataset.model_path,
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        device=device
    )
    
    geometry_filter = GeometryGuidedFilter(
        visual_threshold=0.3,
        geom_threshold=0.05
    )
    
    teacher_student = TeacherStudentTrainer(
        teacher_model=teacher_model,
        student_model=mtgoe_model,
        renderer=renderer,
        geometry_filter=geometry_filter,
        ema_decay=0.999
    )
    
    mtgoe_model.train()
    # pose_model.train()
    
    for epoch in range(start_from_epoch, start_from_epoch + num_epochs):
        epoch_losses = {'total': 0, 'pos': 0, 'depth': 0, 'seg': 0, 'vf': 0}
        valid_batches = 0
        
        progress_bar = tqdm(enumerate(real_loader), total=len(real_loader), desc=f"Epoch {epoch+1}/{start_from_epoch + num_epochs}")
        for batch_idx, batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Train step
            result = teacher_student.train_step(batch, optimizer, multi_task_loss)
            
            if result is not None:
                valid_batches += 1
                loss_val, losses_dict = result
                
                # Update epoch losses
                for k, v in losses_dict.items():
                    epoch_losses[k] += v
                
                # Update progress bar
                avg_loss = epoch_losses['total'] / valid_batches
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}", 'valid_batches': valid_batches})
        
        # Step the scheduler
        scheduler.step()
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{start_from_epoch + num_epochs} Results:")
        for k, v in epoch_losses.items():
            avg_v = v / max(1, valid_batches)
            print(f"  {k}: {avg_v:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            save_checkpoint(epoch)
    
    return start_from_epoch + num_epochs

# Function for evaluation
def evaluate():
    """Evaluate the model on test data"""
    print("\nEvaluating model...")
    mtgoe_model.eval()
    # pose_model.eval()
    
    # Metrics
    translation_errors = []
    rotation_errors = []
    add_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = mtgoe_model(
                batch['rgb'],
                fx=batch['cam_K'][:, 0, 0],
                fy=batch['cam_K'][:, 1, 1],
                px=batch['cam_K'][:, 0, 2],
                py=batch['cam_K'][:, 1, 2]
            )
            
            # Get predictions
            _, centers, translations, _, _, _, _ = outputs
            
            # Get ground truth
            gt_translation = batch['cam_t_m2c'][0] if 'cam_t_m2c' in batch else None
            gt_rotation = batch['cam_R_m2c'][0] if 'cam_R_m2c' in batch else None
            
            if gt_translation is not None and len(translations) > 0:
                # Find the closest prediction to ground truth
                dists = torch.norm(translations - gt_translation.unsqueeze(0), dim=1)
                min_idx = torch.argmin(dists)
                
                # Translation error
                trans_error = dists[min_idx].item()
                translation_errors.append(trans_error)
                
                # Compute ADD metric if we have model points and rotation
                if gt_rotation is not None and hasattr(test_dataset, 'model'):
                    # In real implementation, we would also need to estimate rotation
                    # This is a simplified version that assumes we have ground truth rotation
                    model_points = torch.tensor(test_dataset.model.vertices, 
                                               dtype=torch.float32, device=device)
                    
                    # Transform model points with predicted and ground truth poses
                    pred_points = torch.mm(gt_rotation, model_points.T) + translations[min_idx].view(3, 1)
                    gt_points = torch.mm(gt_rotation, model_points.T) + gt_translation.view(3, 1)
                    
                    # Compute ADD metric
                    model_dists = torch.norm(pred_points - gt_points, dim=0)
                    add = torch.mean(model_dists).item()
                    
                    # Get model diameter
                    if hasattr(batch, 'extens'):
                        diameter = torch.norm(batch['extens'][0]).item()
                    else:
                        # Use a default if not available
                        diameter = 0.1
                    
                    # ADD relative to diameter
                    add_rel = add / diameter
                    add_metrics.append(add_rel)
    
    # Compute average metrics
    avg_trans_error = np.mean(translation_errors) if translation_errors else float('nan')
    avg_add = np.mean(add_metrics) if add_metrics else float('nan')
    
    # Compute ADD accuracy (% of samples with ADD < 10% of diameter)
    add_accuracy = np.mean(np.array(add_metrics) < 0.1) if add_metrics else float('nan')
    
    print(f"Evaluation Results:")
    print(f"  Average Translation Error: {avg_trans_error:.4f} m")
    print(f"  Average ADD: {avg_add:.4f}")
    print(f"  ADD Accuracy (<10%): {add_accuracy:.4f}")
    
    return add_accuracy

# Main training loop
if __name__ == "__main__":
    current_epoch = start_epoch
    
    
    # First supervised training phase
    current_epoch = train_supervised(config['training']['supervised_epochs'])
    
    # Then self-supervised refinement
    current_epoch = train_self_supervised(config['training']['self_supervised_epochs'], current_epoch)
    
    # Final evaluation
    add_accuracy = evaluate()
    
    # Save final model
    save_checkpoint(current_epoch - 1, is_best=(add_accuracy > 0.5))
    
    print("Training and evaluation complete!")
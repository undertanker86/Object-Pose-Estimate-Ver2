import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from model import MTGOE
from train_ver2 import BOPDataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Test MTGOE model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='./data/lmo', help='Path to dataset')
    parser.add_argument('--obj_id', type=int, default=1, help='Object ID to test')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--vis_threshold', type=float, default=0.5, help='Visualization threshold')
    return parser.parse_args()

def save_visualization(image, vote_map, centers, translations, segmentation=None, vector_field=None, depth=None, 
                      gt_centers=None, gt_translations=None, save_path=None):
    """
    Save visualization of model outputs
    
    Args:
        image: Input RGB image (C, H, W)
        vote_map: Vote map from Hough voting
        centers: Detected object centers
        translations: Detected object translations
        segmentation: Segmentation map (optional)
        vector_field: Vector field (optional)
        depth: Depth map (optional)
        gt_centers: Ground truth centers (optional)
        gt_translations: Ground truth translations (optional)
        save_path: Path to save visualization
    """
    # Convert to numpy
    image_np = image.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    vote_map_np = vote_map.detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Plot RGB image with detected centers
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image_np)
    ax1.set_title('RGB Image with Detected Centers')
    
    # Plot centers on RGB image
    for i, center in enumerate(centers):
        cx, cy = center.detach().cpu().numpy()
        ax1.plot(cx, cy, 'ro', markersize=8)
        ax1.text(cx+5, cy+5, f"{i}", color='red', fontsize=12)
    
    # Plot ground truth centers if available
    if gt_centers is not None:
        for i, center in enumerate(gt_centers):
            cx, cy = center.detach().cpu().numpy()
            ax1.plot(cx, cy, 'go', markersize=8)
            ax1.text(cx+5, cy+5, f"GT{i}", color='green', fontsize=12)
    
    # Plot vote map
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(vote_map_np, cmap='jet')
    ax2.set_title('Vote Map')
    plt.colorbar(im, ax=ax2)
    
    # Plot centers on vote map
    for center in centers:
        cx, cy = center.detach().cpu().numpy()
        ax2.plot(cx, cy, 'wo', markersize=8)
    
    # Plot segmentation if available
    if segmentation is not None:
        ax3 = plt.subplot(2, 3, 3)
        if segmentation.shape[0] > 1:  # Multi-class segmentation
            seg_np = torch.argmax(segmentation, dim=0).detach().cpu().numpy()
        else:  # Binary segmentation
            seg_np = segmentation[0].detach().cpu().numpy()
        ax3.imshow(seg_np)
        ax3.set_title('Segmentation')
    
    # Plot depth if available
    if depth is not None:
        ax4 = plt.subplot(2, 3, 4)
        depth_np = depth[0].detach().cpu().numpy()
        im = ax4.imshow(depth_np)
        ax4.set_title('Depth Map')
        plt.colorbar(im, ax=ax4)
    
    # Plot vector field if available
    if vector_field is not None:
        ax5 = plt.subplot(2, 3, 5)
        vf_np = vector_field.detach().cpu().numpy()
        
        # Create a grid of points
        h, w = vf_np.shape[1], vf_np.shape[2]
        stride = 20
        y, x = np.mgrid[0:h:stride, 0:w:stride]
        
        # Get vectors at grid points
        u = vf_np[0, y, x]  # x-component
        v = vf_np[1, y, x]  # y-component
        
        # Plot background image
        ax5.imshow(image_np)
        
        # Plot vector field
        ax5.quiver(x, y, u, v, color='yellow', scale=50)
        ax5.set_title('Vector Field')
    
    # Plot translations in 3D
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    ax6.set_title('3D Translations')
    
    # Plot detected translations
    for i, trans in enumerate(translations):
        tx, ty, tz = trans.detach().cpu().numpy()
        ax6.scatter(tx, ty, tz, c='red', s=100, label=f'Pred {i}' if i == 0 else "")
        ax6.text(tx, ty, tz, f"{i}", color='red', fontsize=12)
    
    # Plot ground truth translations if available
    if gt_translations is not None:
        for i, trans in enumerate(gt_translations):
            tx, ty, tz = trans.detach().cpu().numpy()
            ax6.scatter(tx, ty, tz, c='green', s=100, label=f'GT {i}' if i == 0 else "")
            ax6.text(tx, ty, tz, f"GT{i}", color='green', fontsize=12)
    
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    ax6.legend()
    
    # Set overall title
    plt.suptitle('MTGOE Model Results', fontsize=16)
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test dataset
    test_dataset = BOPDataset(
        dataset_root=args.dataset,
        split='test',
        obj_id=args.obj_id
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Initialize model
    model = MTGOE(
        image_height=test_dataset[0]['rgb'].shape[1],
        image_width=test_dataset[0]['rgb'].shape[2],
        device=device
    ).to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'mtgoe_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['mtgoe_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    else:
        print(f"No checkpoint found at {args.checkpoint}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Metrics
    translation_errors = []
    add_metrics = []
    center_errors = []
    success_count = 0
    
    # Run inference on test data
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Extract camera parameters
            camera_params = {
                'fx': batch['cam_K'][:, 0, 0],
                'fy': batch['cam_K'][:, 1, 1],
                'px': batch['cam_K'][:, 0, 2],
                'py': batch['cam_K'][:, 1, 2]
            }
            
            # Forward pass
            vote_map, centers, translations, depth, seg, vector_field, _ = model(
                batch['rgb'], **camera_params
            )
            
            # Skip if no objects detected
            if len(centers) == 0:
                print(f"No objects detected in batch {batch_idx}")
                continue
            
            success_count += 1
            
            # Calculate metrics if ground truth available
            gt_centers = None
            gt_translations = None
            
            if 'gt_center_x' in batch and 'gt_center_y' in batch:
                gt_centers = torch.stack([batch['gt_center_x'], batch['gt_center_y']], dim=1)
                
                # Calculate center error
                if len(centers) > 0 and gt_centers.shape[0] > 0:
                    min_dist = float('inf')
                    for pred_center in centers:
                        dists = torch.norm(pred_center.unsqueeze(0) - gt_centers, dim=1)
                        batch_min_dist = torch.min(dists).item()
                        min_dist = min(min_dist, batch_min_dist)
                    center_errors.append(min_dist)
            
            if 'cam_t_m2c' in batch:
                gt_translations = batch['cam_t_m2c']
                
                # Calculate translation error
                if len(translations) > 0 and gt_translations.shape[0] > 0:
                    min_dist = float('inf')
                    for pred_trans in translations:
                        dists = torch.norm(pred_trans.unsqueeze(0) - gt_translations, dim=1)
                        batch_min_dist = torch.min(dists).item()
                        min_dist = min(min_dist, batch_min_dist)
                    translation_errors.append(min_dist)
                    
                    # Calculate ADD metric if model available
                    if 'cam_R_m2c' in batch and hasattr(test_dataset, 'model') and test_dataset.model is not None:
                        gt_rotation = batch['cam_R_m2c'][0]
                        gt_translation = gt_translations[0]
                        
                        # Get closest prediction
                        dists = torch.norm(translations - gt_translation.unsqueeze(0), dim=1)
                        min_idx = torch.argmin(dists)
                        pred_translation = translations[min_idx]
                        
                        # Compute ADD metric
                        model_points = torch.tensor(test_dataset.model.vertices, dtype=torch.float32, device=device)
                        pred_points = torch.mm(gt_rotation, model_points.T) + pred_translation.view(3, 1)
                        gt_points = torch.mm(gt_rotation, model_points.T) + gt_translation.view(3, 1)
                        
                        model_dists = torch.norm(pred_points - gt_points, dim=0)
                        add = torch.mean(model_dists).item()
                        
                        # Get model diameter/extent
                        if 'extens' in batch:
                            diameter = torch.norm(batch['extens'][0]).item()
                        else:
                            # Use default if not available
                            diameter = 0.1
                        
                        # ADD relative to diameter
                        add_rel = add / diameter
                        add_metrics.append(add_rel)
            
            # Save visualization
            save_path = os.path.join(args.output_dir, f"result_{batch_idx:04d}.png")
            save_visualization(
                batch['rgb'][0],
                vote_map,
                centers,
                translations,
                segmentation=seg,
                vector_field=vector_field,
                depth=depth,
                gt_centers=gt_centers,
                gt_translations=gt_translations,
                save_path=save_path
            )
    
    # Print metrics
    print(f"\nTest Results:")
    print(f"Objects detected in {success_count} / {len(test_loader)} images")
    
    if center_errors:
        avg_center_error = np.mean(center_errors)
        print(f"Average Center Error: {avg_center_error:.4f} pixels")
    
    if translation_errors:
        avg_trans_error = np.mean(translation_errors)
        print(f"Average Translation Error: {avg_trans_error:.4f} m")
    
    if add_metrics:
        avg_add = np.mean(add_metrics)
        add_accuracy = np.mean(np.array(add_metrics) < 0.1)
        print(f"Average ADD: {avg_add:.4f}")
        print(f"ADD Accuracy (<10%): {add_accuracy:.4f}")

if __name__ == "__main__":
    main()
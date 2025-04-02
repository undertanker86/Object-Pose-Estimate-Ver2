import torch
import torch.nn.functional as F
import numpy as np

class GeometryGuidedFilter:
    def __init__(self, visual_threshold=0.3, geom_threshold=0.05):
        self.visual_threshold = visual_threshold
        self.geom_threshold = geom_threshold
    
    def compute_visual_alignment(self, *args):
        """
        Flexible method to compute visual alignment
        
        Handles different input formats:
        1. (rendered_mask, observed_mask)
        2. (rendered_images, rendered_masks, observed_mask)
        3. (rendered_dict, observed_mask)
        """
        # Case 1: Two arguments 
        if len(args) == 2:
            rendered_mask, observed_mask = args
            # Check if first argument is a dictionary
            if isinstance(rendered_mask, dict):
                rendered_mask = rendered_mask.get('mask')
        
        # Case 2: Three arguments 
        elif len(args) == 3:
            # Ignore the first argument (rendered_images)
            _, rendered_mask, observed_mask = args
        else:
            raise TypeError(f"Unexpected number of arguments: {len(args)}")

        # Convert to tensors if they're not already
        # Extract mask from dictionary if needed
        if isinstance(rendered_mask, dict):
            rendered_mask = rendered_mask.get('mask')
        if isinstance(observed_mask, dict):
            observed_mask = observed_mask.get('mask')

        # Ensure we have valid tensors
        if rendered_mask is None or observed_mask is None:
            raise ValueError("Could not extract valid masks from input")

        # Convert to tensors if they're not already
        if not isinstance(rendered_mask, torch.Tensor):
            rendered_mask = torch.tensor(rendered_mask, dtype=torch.float32)
        if not isinstance(observed_mask, torch.Tensor):
            observed_mask = torch.tensor(observed_mask, dtype=torch.float32)

        # Make sure masks have the same shape
        if rendered_mask.dim() != observed_mask.dim():
            # If dimensions don't match, adjust them
            if rendered_mask.dim() == 3 and observed_mask.dim() == 2:
                observed_mask = observed_mask.unsqueeze(0)  # Now [1, H, W]
            elif rendered_mask.dim() == 2 and observed_mask.dim() == 3:
                rendered_mask = rendered_mask.unsqueeze(0)  # Now [1, H, W]
        
        # Ensure both are float tensors for BCE loss
        rendered_mask = rendered_mask.float()
        observed_mask = observed_mask.float()
        
        # Calculate binary cross-entropy loss
        mask_diff = F.binary_cross_entropy_with_logits(
            rendered_mask,
            observed_mask,
            reduction='mean'
        )
        
        # Convert to alignment score (higher is better)
        alignment_score = 1.0 - mask_diff.item()
        
        return alignment_score
    
    def compute_geometric_alignment(self, predicted_depth, rendered, mask):
        """Calculate geometric alignment score using Chamfer distance"""
        # Extract depth from rendered dictionary if needed
        if isinstance(rendered, dict):
            rendered_depth = rendered['depth']
        else:
            rendered_depth = rendered

        # Only consider pixels where both depths are valid
        valid_mask = (predicted_depth > 0) & (rendered_depth > 0) & mask.bool()
        
        if not valid_mask.any():
            return float('inf')  # No valid pixels for comparison
            
        # Compute Chamfer distance
        pred_depth_valid = predicted_depth[valid_mask]
        render_depth_valid = rendered_depth[valid_mask]
        
        # L1 distance for simplicity
        depth_diff = torch.abs(pred_depth_valid - render_depth_valid).mean()
        
        return depth_diff
        
    def filter_poses(self, predictions, images, depths, renderer):
        """Filter pose predictions based on visual and geometric criteria"""
        filtered_preds = []
        
        for pred in predictions:
            # Render with predicted pose
            try:
                rendered = renderer.render(pred['R'], pred['t'], pred['K'])
                
                # Compute alignment scores
                visual_score = self.compute_visual_alignment(
                    rendered, 
                    pred['mask']
                )
                
                geom_score = self.compute_geometric_alignment(
                    depths[pred['idx']], 
                    rendered, 
                    pred['mask']
                )
                
                print(f"Pose {pred['idx']} - Visual score: {visual_score:.4f}, Geometric score: {geom_score:.4f}")
                
                # Filter based on thresholds
                if visual_score > self.visual_threshold and geom_score < self.geom_threshold:
                    filtered_preds.append(pred)
                    print(f"Pose {pred['idx']} accepted as valid")
                else:
                    print(f"Pose {pred['idx']} rejected (visual threshold: {self.visual_threshold}, geom threshold: {self.geom_threshold})")
            except Exception as e:
                print(f"Error rendering pose {pred['idx']}: {e}")
                
        return filtered_preds


class TeacherStudentTrainer:
    def __init__(self, teacher_model, student_model, renderer, geometry_filter, ema_decay=0.999):
        """
        Initialize the Teacher-Student training framework
        
        Args:
            teacher_model: Teacher model for self-supervision
            student_model: Student model being trained
            renderer: Renderer for generating synthetic views
            geometry_filter: Filter for selecting valid poses
            ema_decay: Exponential moving average decay rate for teacher
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.renderer = renderer
        self.geometry_filter = geometry_filter
        self.ema_decay = ema_decay
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        
        # Initialize EMA for teacher model parameters
        self.update_teacher()
    
    def update_teacher(self):
        """Update teacher model using EMA of student parameters"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), self.student_model.parameters()
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data +
                    (1.0 - self.ema_decay) * student_param.data
                )
    
    def train_step(self, batch, optimizer, loss_fn):
        """
        Execute one training step of the teacher-student self-supervised learning
        
        Args:
            batch: Dictionary of input data
            optimizer: Optimizer for student model
            loss_fn: Loss function for training
            
        Returns:
            tuple: (loss, loss_dict) or None if step was invalid
        """
        # Skip batches with missing data
        if 'rgb' not in batch or 'cam_K' not in batch:
            print("Skipping batch with missing data (rgb or cam_K)")
            return None
        
        # Extract camera parameters
        camera_params = {
            'fx': batch['cam_K'][:, 0, 0],
            'fy': batch['cam_K'][:, 1, 1],
            'px': batch['cam_K'][:, 0, 2],
            'py': batch['cam_K'][:, 1, 2]
        }
        
        # Print camera parameters for debugging
        print(f"Camera parameters: fx={camera_params['fx'].item():.2f}, fy={camera_params['fy'].item():.2f}, "
              f"px={camera_params['px'].item():.2f}, py={camera_params['py'].item():.2f}")
        
        # Step 1: Teacher forward pass (with no gradients)
        with torch.no_grad():
            # Teacher makes predictions on input image
            teacher_outputs = self.teacher_model(batch['rgb'], **camera_params)
            t_vote_map, t_centers, t_translations, t_depth, t_seg, t_vector_field, _ = teacher_outputs
            
            # Debug teacher outputs
            print(f"\nTeacher predictions:")
            print(f"Centers: {len(t_centers)}")
            if len(t_centers) > 0:
                print(f"First center: {t_centers[0]}")
            print(f"Translations: {len(t_translations)}")
            if len(t_translations) > 0:
                print(f"First translation: {t_translations[0]}")
            print(f"Depth: shape={t_depth.shape}, min={t_depth.min().item():.4f}, max={t_depth.max().item():.4f}")
            
            # Skip if teacher didn't find any objects
            if len(t_centers) == 0 or len(t_translations) == 0:
                print("Teacher found no objects - skipping batch")
                return None
            
            # For each detected object, render synthetic view
            valid_poses = []
            
            for idx, (center, translation) in enumerate(zip(t_centers, t_translations)):
                # Skip if center is out of bounds
                if not (0 <= center[0] < batch['rgb'].shape[-1] and 0 <= center[1] < batch['rgb'].shape[-2]):
                    print(f"Center {center} is out of bounds - skipping")
                    continue
                
                # We need to estimate a rotation matrix
                # For simplicity, we'll just use a default orientation looking at the object
                # In a real implementation, you would use a more sophisticated approach
                
                # Default rotation (facing camera)
                cam_R_m2c = torch.eye(3, device=batch['rgb'].device)
                
                # Check if ground truth rotation is available
                if 'cam_R_m2c' in batch:
                    cam_R_m2c = batch['cam_R_m2c'][0]  # Use ground truth rotation from first batch item
                    print(f"Using ground truth rotation from batch")
                
                # Attempt to render with this pose
                try:
                    print(f"Attempting to render with camera matrix: {batch['cam_K'][0].shape}")
                    print(f"Translation: {translation}")
                    print(f"Rotation: {cam_R_m2c}")
                    
                    rendered_output = self.renderer.render(
                        cam_R_m2c, 
                        translation, 
                        batch['cam_K'][0]  # Use camera params from first batch item
                    )
                    
                    print(f"Rendering successful - mask sum: {rendered_output['mask'].sum().item()}")
                    
                    # Store valid pose
                    valid_poses.append({
                        'idx': idx, 
                        'center': center, 
                        'R': cam_R_m2c, 
                        't': translation,
                        'K': batch['cam_K'][0],
                        'rendered': rendered_output,
                        'mask': rendered_output['mask']
                    })
                except Exception as e:
                    print(f"Rendering failed: {e}")
            
            # Skip if no valid poses found
            if not valid_poses:
                print("No valid poses for rendering - skipping batch")
                return None
            
            print(f"Found {len(valid_poses)} valid poses")
            
            # Step 2: Filter poses (optional)
            # Use geometry_filter to select the most promising poses
            # In this implementation, we'll just use the first valid pose
            
            # Create pseudo-ground truth from teacher outputs
            teacher_targets = {
                'depth_map': t_depth,
                'segmentation_mask': torch.argmax(t_seg, dim=1),  # Convert to class indices
                'vector_field': t_vector_field,
                'pose': {
                    'R': torch.stack([pose['R'] for pose in valid_poses]),
                    't': torch.stack([pose['t'] for pose in valid_poses]),
                    'centers': torch.stack([pose['center'] for pose in valid_poses])
                }
            }
            
            # Create rendered targets from first valid pose
            rendered_depth = valid_poses[0]['rendered']['depth']
            rendered_mask = valid_poses[0]['rendered']['mask'] 
            
            # Combine teacher predictions with rendered outputs
            # For depth and mask, prefer rendered output where available
            combined_depth = torch.where(
                rendered_mask > 0.5,
                rendered_depth,
                t_depth
            )
            
            combined_mask = torch.where(
                rendered_mask > 0.5,
                torch.ones_like(t_seg[:, 0:1, :, :]),
                torch.argmax(t_seg, dim=1, keepdim=True).float()
            )
            
            # Use combined outputs as targets for student
            combined_targets = {
                'depth_map': combined_depth,
                'segmentation_mask': combined_mask.squeeze(1),  # Remove channel dim for segmentation
                'vector_field': t_vector_field,  # Keep teacher's vector field
                'pose': teacher_targets['pose']  # Keep teacher's pose information
            }
            
            print("\nTeacher provided targets:")
            for k, v in combined_targets.items():
                if isinstance(v, dict):
                    print(f"{k}: {' '.join([f'{sub_k}={sub_v.shape if isinstance(sub_v, torch.Tensor) else type(sub_v)}' for sub_k, sub_v in v.items()])}")
                elif isinstance(v, torch.Tensor):
                    print(f"{k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}")
        
        # Step 3: Student forward pass (with gradients)
        student_outputs = self.student_model(batch['rgb'], **camera_params)
        
        # Debug student outputs
        s_vote_map, s_centers, s_translations, s_depth, s_seg, s_vector_field, _ = student_outputs
        print(f"\nStudent predictions:")
        print(f"Centers: {len(s_centers)}")
        if len(s_centers) > 0:
            print(f"First center: {s_centers[0]}")
        print(f"Translations: {len(s_translations)}")
        if len(s_translations) > 0:
            print(f"First translation: {s_translations[0]}")
        
        # Step 4: Compute loss and update student
        loss, loss_dict = loss_fn(student_outputs, combined_targets)
        
        # Skip if loss is invalid
        if not torch.isfinite(loss):
            print(f"Warning: Invalid loss value: {loss.item()}")
            return None
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.student_model.parameters()),
            5.0  # Use a reasonable clip value
        )
        optimizer.step()
        
        # Step 5: Update teacher parameters
        self.update_teacher()
        
        return loss, loss_dict
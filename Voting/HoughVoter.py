import torch
import numpy as np
import matplotlib.pyplot as plt

class HoughVoterPyTorch:
    def __init__(self, image_height, image_width, vote_threshold=0.1, device="cuda"):
        """
        Initialize the Hough voting system with PyTorch CUDA support
        
        Args:
            image_height: Height of the input image
            image_width: Width of the input image
            vote_threshold: Threshold for considering a vote valid
            device: Computing device ('cuda' or 'cpu')
        """
        self.height = image_height
        self.width = image_width
        self.vote_threshold = vote_threshold
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"HoughVoter initialized with vote_threshold={vote_threshold} on device={self.device}")
        
        # Camera intrinsic parameters (can be replaced with actual values)
        self.fx = torch.tensor(500.0, device=self.device)  # focal length x
        self.fy = torch.tensor(500.0, device=self.device)  # focal length y 
        self.px = torch.tensor(self.width // 2, device=self.device, dtype=torch.float32)  # principal point x
        self.py = torch.tensor(self.height // 2, device=self.device, dtype=torch.float32)  # principal point y
        
        # Create coordinate grid for vectorized operations
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, device=self.device, dtype=torch.float32), 
            torch.arange(image_width, device=self.device, dtype=torch.float32),
            indexing="ij"
        )
        self.coords = torch.stack([x_coords, y_coords], dim=2)
        
    def cast_votes(self, segmentation_map, center_directions, depth_predictions, fx=500, fy=500, px=320, py=240):
        """
        Cast votes for object centers based on pixel predictions
        
        Args:
            segmentation_map: Binary mask of object class pixels (H x W)
            center_directions: Direction vectors to object center (H x W x 2)
            depth_predictions: Depth predictions for each pixel (H x W)
            
        Returns:
            vote_map: Accumulated voting scores (tensor)
            centers: Detected object centers (tensor)
            translations: 3D translations of detected objects (tensor)
        """
        # Debug original segmentation_map
        print(f"Original segmentation map: shape={segmentation_map.shape}, "
              f"min={segmentation_map.min().item():.4f}, max={segmentation_map.max().item():.4f}")

        # Convert raw logits to probabilities if needed (if segmentation_map has multiple channels)
        if segmentation_map.dim() == 4 and segmentation_map.shape[1] > 1:
            print("Converting multi-class segmentation to binary mask")
            # Take the argmax across classes to get the predicted class
            seg_class = torch.argmax(segmentation_map, dim=1)
            # Convert to binary segmentation (1 for any object, 0 for background)
            # Assuming class 0 is background
            segmentation_map = (seg_class > 0).float()
        elif segmentation_map.dim() == 4 and segmentation_map.shape[1] == 1:
            # If it's a single channel, just squeeze and ensure it's positive
            segmentation_map = segmentation_map.squeeze(1)
            
        # Ensure values are positive
        segmentation_map = torch.clamp(segmentation_map, min=0.0)

        # Debug processed segmentation_map
        print(f"Processed segmentation map: shape={segmentation_map.shape}, "
              f"min={segmentation_map.min().item():.4f}, max={segmentation_map.max().item():.4f}")
        
        # Debug input data
        print(f"HoughVoter inputs - directions: shape={center_directions.shape}, "
              f"min={center_directions.min().item():.2f}, max={center_directions.max().item():.2f}")
        print(f"HoughVoter inputs - depth: shape={depth_predictions.shape}, "
              f"min={depth_predictions.min().item():.2f}, max={depth_predictions.max().item():.2f}")
        
        # Ensure all inputs are on the correct device
        segmentation_map = self._ensure_tensor(segmentation_map)
        center_directions = self._ensure_tensor(center_directions)
        depth_predictions = self._ensure_tensor(depth_predictions)
        
        # Initialize voting map
        vote_map = torch.zeros((self.height, self.width), device=self.device, dtype=torch.float32)
        
        # Update camera parameters in a safer way
        self.fx = fx.clone().detach() if isinstance(fx, torch.Tensor) else torch.tensor(fx, device=self.device)
        self.fy = fy.clone().detach() if isinstance(fy, torch.Tensor) else torch.tensor(fy, device=self.device)
        self.px = px.clone().detach() if isinstance(px, torch.Tensor) else torch.tensor(px, device=self.device)
        self.py = py.clone().detach() if isinstance(py, torch.Tensor) else torch.tensor(py, device=self.device)
        
        # Find pixels belonging to the object class
        object_mask = segmentation_map > 0
        object_indices = torch.nonzero(object_mask, as_tuple=True)
        
        # Debug object pixels
        print(f"Object pixels found: {len(object_indices[0])}")
        
        if len(object_indices[0]) == 0:
            # Return empty tensors if no object pixels
            empty_centers = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            empty_translations = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            return vote_map, empty_centers, empty_translations
        
        # Get coordinates and directions of object pixels
        object_indices = torch.stack(object_indices, dim=1)  # Shape: [N, 2]

        object_coords = self.coords[object_indices[:, 0], object_indices[:, 1]]  # Shape: [N, 2]
        object_directions = center_directions[object_indices[:, 0], object_indices[:, 1]]
        
        # Debug object directions
        print(f"Object directions: shape={object_directions.shape}, "
              f"min_norm={torch.norm(object_directions, dim=1).min().item():.4f}, "
              f"max_norm={torch.norm(object_directions, dim=1).max().item():.4f}")
        
        # Normalize direction vectors - fixed to handle dimensions properly
        dir_magnitudes = torch.norm(object_directions, dim=1, keepdim=True)
        valid_dirs = dir_magnitudes > 1e-5
        normalized_dirs = torch.zeros_like(object_directions)
        
        # Only normalize valid directions (avoid division by zero)
        valid_indices = valid_dirs.squeeze()
        if valid_indices.sum() > 0:  # Make sure there are valid directions
            normalized_dirs[valid_indices] = object_directions[valid_indices] / dir_magnitudes[valid_indices]
            print(f"Normalized directions: min_norm={torch.norm(normalized_dirs, dim=1).min().item():.4f}, "
                  f"max_norm={torch.norm(normalized_dirs, dim=1).max().item():.4f}")
        
        # Debug normalized directions
        print(f"Valid directions: {valid_indices.sum().item()} out of {len(valid_indices)}")
        
        # Vectorized vote casting
        max_dist = 100  # Maximum voting distance
        for t in range(1, max_dist):
            # Compute vote positions for all object pixels at once
            vote_positions = object_coords + normalized_dirs * t
            
            # Round to integers and convert to indices
            vote_x = vote_positions[:, 0].round().long()
            vote_y = vote_positions[:, 1].round().long()
            
            # Filter votes within image boundaries
            valid_votes = (vote_x >= 0) & (vote_x < self.width) & (vote_y >= 0) & (vote_y < self.height)
            
            if valid_votes.sum() > 0:
                valid_x = vote_x[valid_votes]
                valid_y = vote_y[valid_votes]
                
                # Accumulate votes using index_put_ for atomic operations
                vote_map.index_put_((valid_y, valid_x), torch.ones(len(valid_y), device=self.device), accumulate=True)
        
        # Debug vote map
        print(f"Vote map: min={vote_map.min().item():.2f}, max={vote_map.max().item():.2f}, mean={vote_map.mean().item():.2f}")
        
        # Find local maxima in vote map (object centers)
        centers, inlier_maps = self.find_centers(vote_map, segmentation_map, center_directions)
        
        # Debug centers found
        print(f"Centers found: {len(centers)}")
        if len(centers) > 0:
            print(f"First center: {centers[0]}")
        
        # Calculate 3D translations for each center
        translations = self.calculate_translations(centers, inlier_maps, depth_predictions, fx=fx, fy=fy, px=px, py=py)
        
        # Debug translations
        print(f"Translations calculated: {len(translations)}")
        if len(translations) > 0:
            print(f"First translation: {translations[0]}")
        
        return vote_map, centers, translations
    
    def find_centers(self, vote_map, segmentation_map, center_directions):
        """
        Find object centers by detecting local maxima in the voting map
        
        Args:
            vote_map: Accumulated voting scores (tensor)
            segmentation_map: Binary mask of object class pixels (tensor)
            center_directions: Direction vectors to object center (tensor)
            
        Returns:
            centers: Tensor of detected centers (N x 2)
            inlier_maps: Tensor of inlier maps for each center (N x H x W)
        """
        # Apply max pooling for non-maximum suppression
        kernel_size = 3
        padding = kernel_size // 2
        
        pooled = torch.nn.functional.max_pool2d(
            vote_map.unsqueeze(0), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        ).squeeze(0)
        
        # Find local maxima
        is_max = (vote_map == pooled)
        
        # Use adaptive thresholding based on vote_map statistics
        max_value = torch.max(vote_map).item()
        mean_value = torch.mean(vote_map).item()
        
        # More adaptive threshold based on vote map distribution
        if max_value > 0:
            threshold = mean_value + 0.1 * (max_value - mean_value)
            # Ensure threshold is not too high (capture at least some centers)
            threshold = min(threshold, 0.1 * max_value)
        else:
            threshold = 0.1  # Default low threshold if max is 0
        
        print(f"Using adaptive threshold: {threshold:.2f} (max: {max_value:.2f}, mean: {mean_value:.2f})")
        
        is_center = is_max & (vote_map > threshold)
        
        # Get center coordinates
        center_indices = torch.nonzero(is_center, as_tuple=False)  # Returns (N, 2) tensor
        
        if len(center_indices) == 0:
            # Return empty tensors if no centers found
            empty_centers = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
            empty_inlier_maps = torch.zeros((0, self.height, self.width), device=self.device, dtype=torch.bool)
            print("No centers found with current threshold")
            return empty_centers, empty_inlier_maps
        
        # Flip coordinates to (x, y) format for centers
        centers = torch.flip(center_indices, [1]).float()  # Convert to (N, 2) tensor with (x, y) format
        
        # Create inlier maps for all centers
        num_centers = centers.shape[0]
        inlier_maps = torch.zeros((num_centers, self.height, self.width), device=self.device, dtype=torch.bool)
        
        # Find object pixels
        object_indices = torch.nonzero(segmentation_map > 0, as_tuple=False)  # (N, 2) tensor
        
        if len(object_indices) == 0:
            return centers, inlier_maps
        
        # Get coordinates and directions of object pixels
        object_pixels = self.coords[object_indices[:, 0], object_indices[:, 1]]  # (N, 2)
        
        # Get directions - handle different dimension formats
        if center_directions.dim() == 4 and center_directions.shape[1] == 2:
            # Format [B, 2, H, W]
            directions = center_directions[0, :, object_indices[:, 0], object_indices[:, 1]].transpose(0, 1)
        elif center_directions.dim() == 3 and center_directions.shape[-1] == 2:
            # Format [H, W, 2]
            directions = center_directions[object_indices[:, 0], object_indices[:, 1]]
        else:
            raise ValueError(f"Unexpected center_directions shape: {center_directions.shape}")
        
        # Process each center
        print(f"Processing {num_centers} centers with {len(object_indices)} object pixels")
        for i in range(num_centers):
            center_tensor = centers[i].unsqueeze(0)  # (1, 2)
            
            # Compute vectors from pixels to center
            to_center_vectors = center_tensor - object_pixels  # (N, 2)
            
            # Normalize vectors
            to_center_dists = torch.norm(to_center_vectors, dim=1, keepdim=True)
            valid_dists = to_center_dists > 0
            normalized_to_center = torch.zeros_like(to_center_vectors)
            
            # Only normalize valid vectors
            valid_indices_dist = valid_dists.squeeze()
            if valid_indices_dist.sum() > 0:
                normalized_to_center[valid_indices_dist] = to_center_vectors[valid_indices_dist] / to_center_dists[valid_indices_dist]
            
            # Compute dot products - use a more lenient threshold
            dot_products = (normalized_to_center * directions).sum(dim=1)
            
            # Find inliers (dot product > 0.7 instead of 0.9 for more leniency)
            inliers = dot_products > 0.7
            
            # Set inlier map
            if inliers.sum() > 0:
                inlier_indices = object_indices[inliers]
                inlier_maps[i, inlier_indices[:, 0], inlier_indices[:, 1]] = True
                print(f"Center {i} has {inliers.sum().item()} inlier pixels")
        
        return centers, inlier_maps
    
    def calculate_translations(self, centers, inlier_maps, depth_predictions, fx=500, fy=500, px=320, py=240):
        """
        Calculate 3D translations for detected objects
        
        Args:
            centers: Tensor of detected centers (N x 2)
            inlier_maps: Tensor of inlier maps for each center (N x H x W)
            depth_predictions: Depth predictions for each pixel (H x W) or (B x H x W)
            
        Returns:
            translations: Tensor of 3D translations (N x 3)
        """
        num_centers = centers.shape[0]
        translations = torch.zeros((num_centers, 3), device=self.device, dtype=torch.float32)
        
        # Update camera parameters safely
        self.fx = fx.clone().detach() if isinstance(fx, torch.Tensor) else torch.tensor(fx, device=self.device)
        self.fy = fy.clone().detach() if isinstance(fy, torch.Tensor) else torch.tensor(fy, device=self.device)
        self.px = px.clone().detach() if isinstance(px, torch.Tensor) else torch.tensor(px, device=self.device)
        self.py = py.clone().detach() if isinstance(py, torch.Tensor) else torch.tensor(py, device=self.device)
        
        print(f"Camera parameters: fx={self.fx.item():.2f}, fy={self.fy.item():.2f}, px={self.px.item():.2f}, py={self.py.item():.2f}")
        
        # Check and adjust depth_predictions shape if needed
        if len(depth_predictions.shape) == 3 and depth_predictions.shape[0] == 1:
            # If depth has shape [1, H, W], squeeze it to [H, W]
            depth_predictions = depth_predictions.squeeze(0)
        elif len(depth_predictions.shape) == 4 and depth_predictions.shape[0] == 1 and depth_predictions.shape[1] == 1:
            # If depth has shape [1, 1, H, W], squeeze it to [H, W]
            depth_predictions = depth_predictions.squeeze(0).squeeze(0)
        
        print(f"Depth predictions: shape={depth_predictions.shape}, min={depth_predictions.min().item():.4f}, max={depth_predictions.max().item():.4f}")
        
        # ADD THIS CODE HERE - BEGIN
        # Ensure inlier maps match the depth dimensions
        if inlier_maps.shape[1:] != depth_predictions.shape:
            print(f"Resizing inlier maps from {inlier_maps.shape[1:]} to {depth_predictions.shape}")
            import torch.nn.functional as F  # Make sure this import is at the top of your file
            resized_inlier_maps = torch.zeros((inlier_maps.shape[0], *depth_predictions.shape), 
                                            device=self.device, dtype=inlier_maps.dtype)
            for i in range(inlier_maps.shape[0]):
                resized_inlier_maps[i] = F.interpolate(
                    inlier_maps[i].float().unsqueeze(0).unsqueeze(0),
                    size=depth_predictions.shape,
                    mode='nearest'
                ).squeeze(0).squeeze(0).bool()
            inlier_maps = resized_inlier_maps
        # ADD THIS CODE HERE - END
    

        
        for i in range(num_centers):
            cx, cy = centers[i]
            print(f"Processing center {i}: ({cx.item():.1f}, {cy.item():.1f})")
            
            # Method 1: Use center pixel depth directly
            # Get depth at the center position (or nearest pixel)
            x_idx = min(max(0, int(cx.item())), depth_predictions.shape[1] - 1)
            y_idx = min(max(0, int(cy.item())), depth_predictions.shape[0] - 1)
            center_depth = depth_predictions[y_idx, x_idx]
            
            # Method 2: Use average depth from inlier pixels
            inlier_map = inlier_maps[i]
            if inlier_map.sum() > 0:
                inlier_depths = depth_predictions[inlier_map]
                avg_depth = inlier_depths.mean()
                print(f"Center {i} depth: center={center_depth.item():.4f}, avg_inlier={avg_depth.item():.4f} (from {inlier_map.sum().item()} pixels)")
                
                # Use average depth if it's valid and reasonable
                if torch.isfinite(avg_depth) and avg_depth > 0:
                    Tz = avg_depth
                else:
                    Tz = center_depth
            else:
                Tz = center_depth
                print(f"Center {i} depth (no inliers): {center_depth.item():.4f}")
            
            # Ensure depth is positive and finite
            if not torch.isfinite(Tz) or Tz <= 0:
                print(f"Warning: Invalid depth {Tz.item():.4f} for center {i}, using default depth")
                Tz = torch.tensor(1.0, device=self.device)
            
            # Calculate Tx and Ty using projection equation
            Tx = (cx - self.px) * Tz / self.fx
            Ty = (cy - self.py) * Tz / self.fy
            
            translations[i, 0] = Tx
            translations[i, 1] = Ty
            translations[i, 2] = Tz
            
            print(f"Translation {i}: ({Tx.item():.4f}, {Ty.item():.4f}, {Tz.item():.4f})")
        
        return translations
    
    def _ensure_tensor(self, data):
        """Helper function to ensure data is a tensor on the correct device"""
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device=self.device, dtype=torch.float32)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device).float()
        else:
            raise TypeError(f"Expected numpy array or torch tensor, got {type(data)}")

    def visualize_results(self, vote_map, centers, translations, segmentation_map=None, center_directions=None):
        """
        Visualize the results of Hough voting
        
        Args:
            vote_map: Accumulated voting scores
            centers: Detected object centers
            translations: 3D translations of detected objects
            segmentation_map: Binary mask of object class pixels
            center_directions: Direction vectors to object center
        """
        # Convert tensors to numpy for visualization
        vote_map_np = vote_map.detach().cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        translations_np = translations.detach().cpu().numpy()
        
        if segmentation_map is not None:
            segmentation_np = segmentation_map.detach().cpu().numpy()
        
        if center_directions is not None:
            directions_np = center_directions.detach().cpu().numpy()
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original segmentation
        if segmentation_map is not None:
            axes[0].imshow(segmentation_np, cmap='gray')
            axes[0].set_title('Object Segmentation')
        
        # Direction vectors (subsampled for clarity)
        if segmentation_map is not None and center_directions is not None:
            axes[1].imshow(segmentation_np, cmap='gray')
            stride = 20
            y_indices, x_indices = np.where(segmentation_np > 0)
            for i in range(0, len(y_indices), stride):
                y, x = y_indices[i], x_indices[i]
                nx, ny = directions_np[y, x]
                axes[1].arrow(x, y, nx*15, ny*15, head_width=3, head_length=3, fc='red', ec='red')
            axes[1].set_title('Center Direction Vectors')
        
        # Voting map and detected centers
        axes[2].imshow(vote_map_np, cmap='jet')
        axes[2].set_title('Vote Map and Detected Centers')
        for center in centers_np:
            cx, cy = center
            axes[2].plot(cx, cy, 'wo', markersize=10)
            axes[2].plot(cx, cy, 'ko', markersize=8)
        
        plt.tight_layout()
        plt.show()
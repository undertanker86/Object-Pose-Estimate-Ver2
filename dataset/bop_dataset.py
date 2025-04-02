import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
from collections import defaultdict
import trimesh
from model import MTGOE
import torch
from pprint import pprint
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from TeacherRenderer import TeacherRenderer

class BOPDataset(Dataset):
    def __init__(self, dataset_root, split='train', obj_id=1, transforms=None):
        self.dataset_root = dataset_root
        self.split = split
        self.obj_id = obj_id
        self.transforms = transforms
        self.data = []  # Store data in a list of dictionaries
        
        # Load 3D model
        model_path = os.path.join(self.dataset_root, 'models', f'obj_{self.obj_id:06d}.ply')
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = self.load_ply_model(model_path)
        else:
            print(f"Warning: No model found at {model_path}")
            self.model = None
        
        # Load global camera parameters if available
        camera_path = os.path.join(self.dataset_root, 'camera.json')
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                self.global_camera_params = json.load(f)
        else:
            self.global_camera_params = None
        
        # Path to the split (train or test)
        split_path = os.path.join(self.dataset_root, self.split)
        if not os.path.exists(split_path):
            print(f"Warning: Split path {split_path} does not exist")
            return
        
        # Iterate over scene folders in the split
        for scene_folder in os.listdir(split_path):
            scene_path = os.path.join(split_path, scene_folder)
            if not os.path.isdir(scene_path):
                continue
                
            # Load scene ground truth
            scene_gt_path = os.path.join(scene_path, 'scene_gt.json')
            if not os.path.exists(scene_gt_path):
                print(f"Warning: No scene_gt.json found in {scene_path}")
                continue
                
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)
                
            # Load scene camera parameters
            scene_camera_path = os.path.join(scene_path, 'scene_camera.json')
            if not os.path.exists(scene_camera_path):
                print(f"Warning: No scene_camera.json found in {scene_path}")
                continue
                
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # Load gt info if available
            gt_info = {}
            gt_info_path = os.path.join(scene_path, 'scene_gt_info.json') 
            if os.path.exists(gt_info_path):
                with open(gt_info_path, 'r') as f:
                    gt_info = json.load(f)

            # Iterate over images in this scene
            for img_id_str, objects in scene_gt.items():
                img_id = int(img_id_str)
                
                # Check if camera parameters exist for this image
                if img_id_str not in scene_camera:
                    print(f"Warning: No camera parameters for image {img_id} in scene {scene_folder}")
                    continue
                
                cam_params = scene_camera[img_id_str]
                
                # Path to RGB image
                rgb_path = os.path.join(scene_path, 'rgb', f'{img_id:06d}.png')
                if not os.path.exists(rgb_path):
                    print(f"Warning: RGB image not found at {rgb_path}")
                    continue
                
                # Find our object of interest in this image
                for obj_idx, obj_info in enumerate(objects):
                    if obj_info['obj_id'] == self.obj_id:
                        # Paths to mask files
                        mask_path = os.path.join(scene_path, 'mask', f'{img_id:06d}_{obj_idx:06d}.png')
                        mask_visib_path = os.path.join(scene_path, 'mask_visib', f'{img_id:06d}_{obj_idx:06d}.png')
                        
                        # Check if mask files exist
                        mask_exists = os.path.exists(mask_path)
                        mask_visib_exists = os.path.exists(mask_visib_path)
                        
                        # Initialize bbox values
                        gt_center_x, gt_center_y, visib_fract = 0, 0, 0
                        
                        # Get bbox info if available
                        if img_id_str in gt_info:
                            img_gt_info = gt_info[img_id_str]
                            # Check if gt_info contains a list of objects
                            if isinstance(img_gt_info, list) and len(img_gt_info) > obj_idx:
                                obj_gt_info = img_gt_info[obj_idx]
                                if "bbox_obj" in obj_gt_info and "visib_fract" in obj_gt_info:
                                    bbox_obj = obj_gt_info["bbox_obj"]
                                    visib_fract = obj_gt_info["visib_fract"]
                                    gt_center_x = bbox_obj[0] + bbox_obj[2] / 2
                                    gt_center_y = bbox_obj[1] + bbox_obj[3] / 2
                            # If gt_info is a dictionary (for single object)
                            elif isinstance(img_gt_info, dict):
                                if "bbox_obj" in img_gt_info and "visib_fract" in img_gt_info:
                                    bbox_obj = img_gt_info["bbox_obj"]
                                    visib_fract = img_gt_info["visib_fract"]
                                    gt_center_x = bbox_obj[0] + bbox_obj[2] / 2
                                    gt_center_y = bbox_obj[1] + bbox_obj[3] / 2
                        
                        # Add sample to dataset
                        self.data.append({
                            'scene_id': scene_folder,
                            'img_id': img_id,
                            'obj_idx': obj_idx,
                            'obj_id': self.obj_id,
                            'rgb_path': rgb_path,
                            'mask_path': mask_path if mask_exists else None,
                            'mask_visib_path': mask_visib_path if mask_visib_exists else None,
                            'cam_R_m2c': obj_info.get('cam_R_m2c', None),
                            'cam_t_m2c': obj_info.get('cam_t_m2c', None),
                            'cam_K': cam_params.get('cam_K', None),
                            'gt_center_x': gt_center_x,
                            'gt_center_y': gt_center_y,
                            'visib_fract': visib_fract,
                        })
        
        print(f"Loaded {len(self.data)} samples for object {obj_id} in {split} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load RGB image
        rgb = cv2.imread(sample['rgb_path'])
        if rgb is None:
            raise ValueError(f"Could not load RGB image from {sample['rgb_path']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        if sample['mask_path'] and os.path.exists(sample['mask_path']):
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            # Normalize mask (sometimes BOP masks are 255 for object pixels)
            mask = (mask > 0).astype(np.uint8)
        
        # Load visible mask if available
        mask_visib = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        if sample['mask_visib_path'] and os.path.exists(sample['mask_visib_path']):
            mask_visib = cv2.imread(sample['mask_visib_path'], cv2.IMREAD_GRAYSCALE)
            # Normalize mask (sometimes BOP masks are 255 for object pixels)
            mask_visib = (mask_visib > 0).astype(np.uint8)
        
        # Convert to torch tensors
        rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # Normalize to [0, 1]
        mask_tensor = torch.from_numpy(mask).float()
        mask_visib_tensor = torch.from_numpy(mask_visib).float()

        gt_center_x = sample['gt_center_x']
        gt_center_y = sample['gt_center_y']
        visib_fract = sample['visib_fract']

        gt_center_x_tensor = torch.tensor(gt_center_x, dtype=torch.float32)
        gt_center_y_tensor = torch.tensor(gt_center_y, dtype=torch.float32)
        visib_fract_tensor = torch.tensor(visib_fract, dtype=torch.float32)

        # Get camera parameters
        cam_K = np.array(sample['cam_K']).reshape(3, 3)
        cam_K_tensor = torch.from_numpy(cam_K).float()
        
        # Get pose if available
        cam_R_m2c = sample.get('cam_R_m2c', None)
        cam_t_m2c = sample.get('cam_t_m2c', None)
        
        cam_R_m2c_tensor = torch.from_numpy(np.array(cam_R_m2c).reshape(3, 3)).float() if cam_R_m2c else None
        cam_t_m2c_tensor = torch.from_numpy(np.array(cam_t_m2c)).float() if cam_t_m2c else None
        
        # Create depth map if the model and pose are available
        depth_map_tensor = None
        if self.model is not None and cam_R_m2c is not None and cam_t_m2c is not None:
            depth_map = self.generate_depth_map(cam_R_m2c, cam_t_m2c, cam_K, rgb.shape[0], rgb.shape[1])
            depth_map_tensor = torch.from_numpy(depth_map).float()
        
        # Create vector field if the model and pose are available
        vector_field_tensor = None
        if self.model is not None and cam_R_m2c is not None and cam_t_m2c is not None:
            vector_field = self.generate_field_vectors(cam_R_m2c, cam_t_m2c, self.model, cam_K, rgb.shape[0], rgb.shape[1])
            vector_field_tensor = torch.from_numpy(vector_field).float()
        
        # Calculate model extent/diameter for metrics
        extens = None
        if self.model is not None:
            extens = self.compute_extent(self.model_path)
            extens = torch.tensor(extens, dtype=torch.float32)
        
        # Create return dictionary
        result = {
            'rgb': rgb_tensor,
            'mask': mask_tensor,
            'mask_visib': mask_visib_tensor,
            'cam_K': cam_K_tensor,
            'obj_id': sample['obj_id'],
            'img_id': sample['img_id'],
            'gt_center_x': gt_center_x_tensor,
            'gt_center_y': gt_center_y_tensor,
            'visib_fract': visib_fract_tensor
        }
        
        # Add optional fields if available
        if cam_R_m2c_tensor is not None:
            result['cam_R_m2c'] = cam_R_m2c_tensor
        if cam_t_m2c_tensor is not None:
            result['cam_t_m2c'] = cam_t_m2c_tensor
        if depth_map_tensor is not None:
            result['depth_map'] = depth_map_tensor
        if vector_field_tensor is not None:
            result['vector_field'] = vector_field_tensor
        if mask_tensor is not None:
            result['segmentation_mask'] = mask_tensor  # Using the same mask for segmentation
        if extens is not None:
            result['extens'] = extens
        
        # Apply transforms if specified
        if self.transforms:
            result = self.transforms(result)
        
        return result
        
    @staticmethod
    def load_ply_model(ply_path):
        """Loads a 3D model from a PLY file."""
        try:
            model = trimesh.load_mesh(ply_path)
            return model
        except Exception as e:
            print(f"Error loading PLY model from {ply_path}: {e}")
            return None
            
    def generate_depth_map(self, R, t, K, height, width):
        """Generates a depth map from the object's pose and 3D model."""
        if self.model is None:
            return np.zeros((height, width), dtype=np.float32)
            
        # Convert inputs to numpy arrays if they're not already
        R = np.array(R).reshape(3, 3)
        t = np.array(t)
        K = np.array(K).reshape(3, 3)
        
        # Get model vertices
        points_model = np.array(self.model.vertices)
        
        # Transform to camera coordinates
        points_camera = np.dot(points_model, R.T) + t
        
        # Project to image coordinates
        points_image = np.dot(points_camera, K.T)
        
        # Normalize
        z = points_image[:, 2]
        points_image[:, 0] /= z
        points_image[:, 1] /= z
        
        # Create depth map
        depth_map = np.zeros((height, width), dtype=np.float32)
        
        # Filter points that are in the image bounds
        valid_idx = (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & \
                    (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
        
        # Convert to integer pixel coordinates
        x = np.round(points_image[valid_idx, 0]).astype(int)
        y = np.round(points_image[valid_idx, 1]).astype(int)
        z_valid = z[valid_idx]
        
        # Assign depth values (handle z-fighting with minimum depth)
        for i in range(len(x)):
            if depth_map[y[i], x[i]] == 0 or z_valid[i] < depth_map[y[i], x[i]]:
                depth_map[y[i], x[i]] = z_valid[i]
        
        return depth_map
    
    def generate_field_vectors(self, R, t, model, K, height, width):
        """Generates a vector field pointing to the object center."""
        # Create empty vector field
        vector_field = np.zeros((height, width, 2), dtype=np.float32)
        
        # Convert to numpy arrays
        R = np.array(R).reshape(3, 3)
        t = np.array(t)
        K = np.array(K).reshape(3, 3)
        
        # Compute object center in 3D
        vertices = np.array(model.vertices)
        center_3d = vertices.mean(axis=0)
        
        # Transform center to camera coordinates
        center_camera = np.dot(center_3d, R.T) + t
        
        # Project center to image
        center_image = np.dot(center_camera, K.T)
        center_image /= center_image[2]
        center_x, center_y = center_image[0], center_image[1]
        
        # Generate vector field (vectors pointing to center)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Compute vectors (from pixel to center)
        dx = center_x - x_coords
        dy = center_y - y_coords
        
        # Normalize vectors
        magnitude = np.sqrt(dx**2 + dy**2) + 1e-10
        dx /= magnitude
        dy /= magnitude
        
        # Store in vector field
        vector_field[:, :, 0] = dx
        vector_field[:, :, 1] = dy
        
        return vector_field
    
    @staticmethod
    def compute_extent(model_path):
        """Compute the extent (width, height, depth) of the 3D model."""
        try:
            mesh = trimesh.load_mesh(model_path)
            min_corner = mesh.bounds[0]  # Minimum XYZ coordinates
            max_corner = mesh.bounds[1]  # Maximum XYZ coordinates
            extent = max_corner - min_corner  # Compute width, height, depth
            return extent
        except Exception as e:
            print(f"Error computing extent from {model_path}: {e}")
            return np.array([0.1, 0.1, 0.1])  # Default extent

if __name__ == '__main__':
    print(torch.cuda.memory_summary())


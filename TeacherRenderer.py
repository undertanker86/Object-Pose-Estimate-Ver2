import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import pyrender
import math
import os

class TeacherRenderer(nn.Module):
    def __init__(self, cad_model_path, image_height, image_width, device='cuda'):
        super(TeacherRenderer, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        
        # Load CAD model
        self.mesh = trimesh.load(cad_model_path)
        self.vertices = torch.tensor(self.mesh.vertices, dtype=torch.float32, device=device)
        if hasattr(self.mesh, 'faces'):
            self.faces = torch.tensor(self.mesh.faces, dtype=torch.int64, device=device)
        else:
            print("Warning: Mesh has no faces. Using point cloud rendering.")
            self.faces = None
        
        # Setup renderer
        self.setup_renderer()
        
    def setup_renderer(self):
        """Set up the pyrender offscreen renderer with appropriate backend"""
        # Configure pyrender to use offscreen rendering
        # Try different backends in order of preference
        try:
            # First try EGL (works well with NVIDIA GPUs)
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            self.renderer = pyrender.OffscreenRenderer(self.image_width, self.image_height)
        except Exception as e:
            print(f"EGL initialization failed: {e}")
            try:
                # Then try OSMesa (software rendering)
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                self.renderer = pyrender.OffscreenRenderer(self.image_width, self.image_height)
            except Exception as e:
                print(f"OSMesa initialization failed: {e}")
                # As a last resort, try a dummy display
                os.environ.pop('PYOPENGL_PLATFORM', None)  # Clear previous setting
                os.environ['DISPLAY'] = ':0'  # Try default display
                self.renderer = pyrender.OffscreenRenderer(self.image_width, self.image_height)
        
        self.scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
        
        # Add mesh to scene
        mesh = pyrender.Mesh.from_trimesh(self.mesh)
        self.mesh_node = self.scene.add(mesh)
        
        # Add camera to scene - don't specify yfov yet, will update per render
        camera = pyrender.PerspectiveCamera(yfov=math.pi / 3.0)
        self.camera_node = self.scene.add(camera)
        
        print("Renderer successfully initialized")
        
    def render(self, cam_R_m2c, cam_t_m2c, cam_K):
        """
        Render the object with the given pose.
        
        Args:
            cam_R_m2c (Tensor): Rotation matrix [3, 3]
            cam_t_m2c (Tensor): Translation vector [3]
            cam_K (Tensor): Camera intrinsic matrix [3, 3]
            
        Returns:
            dict: Rendered outputs including RGB, depth, and mask
        """
        # Convert to numpy for pyrender
        R = cam_R_m2c.detach().cpu().numpy()
        t = cam_t_m2c.detach().cpu().numpy()
        K = cam_K.detach().cpu().numpy()
        
        # Create camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = t
        
        # Update camera pose
        self.scene.set_pose(self.camera_node, camera_pose)
        
        # Get the camera
        camera_nodes = [node for node in self.scene.nodes if node.camera is not None]
        if not camera_nodes:
            raise ValueError("No camera found in scene")
        
        # Extract intrinsic parameters
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        # Calculate field of view from focal length and image dimensions
        yfov = 2 * math.atan(self.image_height / (2 * fy))
        
        # Create new camera
        new_camera = pyrender.PerspectiveCamera(
            yfov=yfov,
            aspectRatio=fx/fy,
            znear=0.01,
            zfar=100.0
        )
        
        # Remove old camera and add new one with same pose
        old_camera_node = camera_nodes[0]
        self.scene.remove_node(old_camera_node)
        self.camera_node = self.scene.add(new_camera, pose=camera_pose)
        
        # Render
        rgb, depth = self.renderer.render(self.scene)
        
        # Fix the negative stride issue by making a copy of the arrays
        rgb = np.copy(rgb)
        depth = np.copy(depth)
        
        # Create mask
        mask = (depth > 0).astype(np.float32)
        
        # Convert to torch tensors
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1) / 255.0
        depth_tensor = torch.tensor(depth, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor
        }
        
    def __del__(self):
        # Clean up renderer resources
        if hasattr(self, 'renderer'):
            self.renderer.delete()
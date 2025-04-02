import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from Voting.HoughVoter import HoughVoterPyTorch

class MultiEmbedding(nn.Module):
    def __init__(self, in_channels, num_head=4, ratio=2):
        super(MultiEmbedding, self).__init__()
        self.num_head = num_head
        self.head_dim = in_channels // num_head
        self.scale = self.head_dim ** -0.5
        
        # Project queries, keys, and values
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, ref):
        """
        x: Target features to be enhanced
        ref: Reference features to provide context
        """
        B, C, H, W = x.shape
        
        # Reshape for multi-head attention
        q = self.q_proj(x).view(B, self.num_head, self.head_dim, H*W)
        k = self.k_proj(ref).view(B, self.num_head, self.head_dim, H*W)
        v = self.v_proj(ref).view(B, self.num_head, self.head_dim, H*W)
        
        # Transpose for matrix multiplication
        q = q.transpose(2, 3)  # [B, num_head, H*W, head_dim]
        k = k.transpose(2, 3)  # [B, num_head, H*W, head_dim]
        v = v.transpose(2, 3)  # [B, num_head, H*W, head_dim]
        
        # Attention
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale  # [B, num_head, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_head, H*W, head_dim]
        out = out.transpose(2, 3).reshape(B, C, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection
        return x + out

class CMAModule(nn.Module):
    def __init__(self, channels, num_head=4, ratio=2):
        super(CMAModule, self).__init__()
        self.depth_to_seg = MultiEmbedding(channels, num_head, ratio)
        self.seg_to_depth = MultiEmbedding(channels, num_head, ratio)
        
    def forward(self, depth_features, seg_features):
        # Enhance depth features with segmentation information
        enhanced_depth = self.seg_to_depth(depth_features, seg_features)
        
        # Enhance segmentation features with depth information
        enhanced_seg = self.depth_to_seg(seg_features, depth_features)
        
        return enhanced_depth, enhanced_seg

class MTGOE(nn.Module):
    def __init__(self, image_height=480, image_width=640, pretrained=True, device='cuda', cma_layers=[2, 1]):
        super(MTGOE, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.device = device
        self.cma_layers = cma_layers
        
        # Feature dimensions for MobileNetV2
        # MobileNetV2 feature dimensions at different layers
        self.num_ch_enc = [16, 24, 32, 96, 1280]  # Approximate values from MobileNetV2
        
        # Initialize encoder with pre-trained MobileNetV2
        self.encoder = self.build_encoder(pretrained)
        
        # Decoder channel dimensions
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        # Build CMA modules for specified layers
        self.cma_modules = nn.ModuleDict()
        for layer in cma_layers:
            self.cma_modules[f"layer{layer}"] = CMAModule(
                self.num_ch_dec[layer], 
                num_head=4, 
                ratio=2
            )
        
        # Build decoders
        self.depth_decoders = self.build_depth_decoder()
        self.seg_decoders = self.build_seg_decoder()
        self.vf_decoders = self.build_vf_decoder()
        
        # Initialize Multi-Modal Distillation module
        self.mmd = self.build_mmd()
        
        # Initialize HoughVoter for center detection
        self.voter = HoughVoterPyTorch(
            image_height=image_height, 
            image_width=image_width,
            vote_threshold=0.1,
            device=device
        )
    
    def build_encoder(self, pretrained=True):
        # Use MobileNetV2 as the encoder
        mobilenet = mobilenet_v2(pretrained=pretrained)
        
        # Extract the features from MobileNetV2
        features = mobilenet.features
        
        # We keep the model up to the final activation
        return features
    
    def build_depth_decoder(self):
        decoders = nn.ModuleList()
        
        # Build decoder blocks from highest to lowest resolution
        # Start from the output of the encoder (highest level features)
        in_channels = self.num_ch_enc[-1]
        
        for i in range(5):
            out_channels = self.num_ch_dec[4-i]
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            decoders.append(decoder_block)
            in_channels = out_channels
        
        # Final output layer for depth
        decoders.append(nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=1))
        
        return decoders
    
    def build_seg_decoder(self):
        decoders = nn.ModuleList()
        
        # Build decoder blocks from highest to lowest resolution
        in_channels = self.num_ch_enc[-1]
        
        for i in range(5):
            out_channels = self.num_ch_dec[4-i]
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            decoders.append(decoder_block)
            in_channels = out_channels
        
        # Final output layer for segmentation
        decoders.append(nn.Conv2d(self.num_ch_dec[0], 19, kernel_size=1))
        
        return decoders
    
    def build_vf_decoder(self):
        decoders = nn.ModuleList()
        
        # Build decoder blocks from highest to lowest resolution
        in_channels = self.num_ch_enc[-1]
        
        for i in range(5):
            out_channels = self.num_ch_dec[4-i]
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )
            decoders.append(decoder_block)
            in_channels = out_channels
        
        # Final output layer for vector field
        decoders.append(nn.Conv2d(self.num_ch_dec[0], 2, kernel_size=1))
        
        return decoders
    
    def build_mmd(self):
        # Multi-Modal Distillation module
        return nn.Sequential(
            nn.Conv2d(19 + 1 + 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, fx=None, fy=None, px=None, py=None):
        # Set default camera parameters if not provided
        if fx is None:
            fx = torch.tensor(self.image_width / 2, device=x.device)
        if fy is None:
            fy = torch.tensor(self.image_height / 2, device=x.device)
        if px is None:
            px = torch.tensor(self.image_width / 2, device=x.device)
        if py is None:
            py = torch.tensor(self.image_height / 2, device=x.device)
        
        # Encode input image
        encoded = self.encoder(x)
        
        # Initialize decoder features
        depth_features = encoded
        seg_features = encoded
        vf_features = encoded
        
        # Decoder levels (from highest to lowest)
        depth_level_features = []
        seg_level_features = []
        
        # Process through decoder layers
        for i in range(5):
            # Apply depth decoder blocks
            depth_features = self.depth_decoders[i](depth_features)
            depth_level_features.append(depth_features)
            
            # Apply segmentation decoder blocks
            seg_features = self.seg_decoders[i](seg_features)
            seg_level_features.append(seg_features)
            
            # Apply vector field decoder blocks
            vf_features = self.vf_decoders[i](vf_features)
            
            # Apply CMA at specified layers
            if 4-i in self.cma_layers:
                cma_module = self.cma_modules[f"layer{4-i}"]
                depth_features, seg_features = cma_module(depth_features, seg_features)
                # Update the level features after CMA
                depth_level_features[-1] = depth_features
                seg_level_features[-1] = seg_features
        
        # Final output layers
        depth = self.depth_decoders[-1](depth_features)
        seg = self.seg_decoders[-1](seg_features)
        vector_field = self.vf_decoders[-1](vf_features)
        
        # Print intermediate output stats
        print(f"Raw seg output: shape={seg.shape}, min={seg.min().item():.4f}, max={seg.max().item():.4f}")
        print(f"Raw depth output: shape={depth.shape}, min={depth.min().item():.4f}, max={depth.max().item():.4f}")
        print(f"Raw vector field output: shape={vector_field.shape}, min={vector_field.min().item():.4f}, max={vector_field.max().item():.4f}")
        
        # Normalize vector field to unit vectors
        vector_magnitude = torch.norm(vector_field, dim=1, keepdim=True) + 1e-8
        normalized_vector_field = vector_field / vector_magnitude
        vector_field = normalized_vector_field
        
        print(f"Normalized vector field stats: min={vector_field.min().item():.4f}, max={vector_field.max().item():.4f}, mean={vector_field.mean().item():.4f}")
        
        # Generate binary segmentation (0: background, 1: object)
        seg_probs = F.softmax(seg, dim=1)  # Convert logits to probabilities
        seg_binary = (torch.argmax(seg_probs, dim=1) > 0).float()  # Create binary segmentation
        
        print(f"Binary segmentation: shape={seg_binary.shape}, sum={seg_binary.sum().item():.2f}")
        
        # Concatenate task outputs for multi-modal distillation
        features = torch.cat([seg, depth, vector_field], dim=1)
        features = self.mmd(features)
        
        # Generate vote map, centers, and translations using HoughVoter
        vote_map, centers, translations = self.voter.cast_votes(
            seg_binary,  # Use binary segmentation 
            vector_field.permute(0, 2, 3, 1).squeeze(0),  # Format as [H, W, 2]
            depth.squeeze(1),  # Format as [B, H, W]
            fx=fx, 
            fy=fy, 
            px=px, 
            py=py
        )
        
        print(f"Model forward: detected {len(centers)} centers and {len(translations)} translations")
        if len(centers) > 0:
            print(f"First center: {centers[0]}")
        if len(translations) > 0:  
            print(f"First translation: {translations[0]}")
        
        # Return the outputs
        return vote_map, centers, translations, depth, seg, vector_field, features
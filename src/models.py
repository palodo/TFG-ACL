"""
Models module for ACL Classifier
Includes CNN Selector, ViT classifier, CNN baseline, and loss functions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import ViTModel
from torchvision import models


# ==========================================
# CNN SELECTOR CLASSES
# ==========================================

class ACLSliceClassifier(nn.Module):
    """
    CNN that classifies if a single slice contains visible ACL.
    Architecture: MobileNetV2 (pretrained) + custom classifier
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv2_100', pretrained=pretrained, num_classes=0)
        
        # Custom classifier (same as used in original training)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # 2 classes: [no ACL, with ACL]
        )
    
    def forward(self, x):
        return self.backbone(x)


class SimpleCNNSelector(nn.Module):
    """
    Wrapper for CNN Selector for production use.
    Loads trained model and provides slice selection methods.
    Automatically handles different checkpoint formats.
    """
    def __init__(self, checkpoint_path, device='cuda:0'):
        super().__init__()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Clean names (in case they come from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # IMPROVED DETECTION: Handle different checkpoint formats
        has_backbone_prefix = any(k.startswith('backbone.') for k in new_state_dict.keys())
        has_classifier_only = any(k.startswith('classifier.') and not k.startswith('backbone.') 
                                 for k in new_state_dict.keys())
        
        # If checkpoint has 'classifier.' but not 'backbone.classifier.', add prefix
        if has_classifier_only and not has_backbone_prefix:
            print(f"  ⚙️  Detectado formato de checkpoint sin prefijo 'backbone.', remapeando keys...")
            remapped_state_dict = {}
            for k, v in new_state_dict.items():
                if k.startswith('classifier.'):
                    new_key = 'backbone.' + k
                    remapped_state_dict[new_key] = v
                else:
                    remapped_state_dict[k] = v
            new_state_dict = remapped_state_dict
            has_backbone_prefix = True
        
        # Determine architecture
        has_sequential_architecture = (
            ('backbone.classifier.1.weight' in new_state_dict) or 
            ('classifier.1.weight' in new_state_dict)
        )
        
        if has_sequential_architecture:
            # Sequential architecture with direct Linear in position 1
            class SequentialACLClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = timm.create_model('mobilenetv2_100', pretrained=False, num_classes=0)
                    self.backbone.classifier = nn.Sequential(
                        nn.Dropout(0.2),
                        nn.Linear(1280, 2)
                    )
                
                def forward(self, x):
                    return self.backbone(x)
            
            self.model = SequentialACLClassifier()
        else:
            # Full architecture (5 layers in classifier)
            self.model = ACLSliceClassifier(pretrained=False)
        
        # Load weights
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️  Keys faltantes en checkpoint: {missing_keys[:3]}...")
        if unexpected_keys:
            print(f"  ℹ️  Keys inesperadas en checkpoint: {unexpected_keys[:3]}...")
        
        self.model.eval()
        
        # Move to specified GPU
        self.device = device
        self.model = self.model.to(device)
        
        print(f"✓ CNN Selector cargado desde: {checkpoint_path}")
        if 'val_f1' in checkpoint:
            print(f"  Validation F1: {checkpoint['val_f1']:.4f}")
        print(f"  Device: {device}")
    
    @torch.no_grad()
    def get_acl_probability(self, slice_2d):
        """
        Compute probability that a slice contains ACL.
        
        Args:
            slice_2d: numpy array (H, W) normalized [0, 1]
        
        Returns:
            float: Probability [0, 1] of ACL presence
        """
        # Convert to tensor and replicate to 3 channels (RGB)
        slice_tensor = torch.from_numpy(slice_2d).float()
        slice_tensor = slice_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        slice_tensor = slice_tensor.repeat(1, 3, 1, 1)  # (1, 3, H, W)
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        slice_tensor = (slice_tensor - mean) / std
        
        # Move to GPU and predict
        slice_tensor = slice_tensor.to(self.device)
        logits = self.model(slice_tensor)
        probs = torch.softmax(logits, dim=1)
        
        # Return probability of class 1 (contains ACL)
        return probs[0, 1].item()
    
    def select_slices_batch(self, volumes, k=5):
        """
        Select K most informative slices for a batch of volumes.
        OPTIMIZED: Processes multiple volumes in parallel.
        
        Args:
            volumes: numpy array (B, num_slices, H, W)
            k: number of slices to select per volume
        
        Returns:
            list: List of K selected indices for each volume
        """
        batch_size, num_slices, H, W = volumes.shape
        all_indices = []
        
        # Process each volume in batch
        for b in range(batch_size):
            volume = volumes[b]  # (num_slices, H, W)
            
            # Compute ACL probability for each slice
            acl_probs = []
            for i in range(num_slices):
                prob = self.get_acl_probability(volume[i])
                acl_probs.append(prob)
            
            # Select K slices with highest probability
            indices = np.argsort(acl_probs)[-k:]
            indices = sorted(indices.tolist())
            all_indices.append(indices)
        
        return all_indices


# ==========================================
# VIT CLASSIFIER CLASSES
# ==========================================

class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which slices are most important.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, feature_dim)
        Returns:
            weighted_features: tensor (batch_size, feature_dim)
            attention_weights: tensor (batch_size, num_slices)
        """
        # Compute attention scores for each slice
        attn_scores = self.attention(x)  # (batch_size, num_slices, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # Softmax normalization
        
        # Weighted average
        weighted_features = (x * attn_weights).sum(dim=1)  # (batch_size, feature_dim)
        
        return weighted_features, attn_weights.squeeze(-1)


class ViTMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with ViT.
    
    Architecture:
    1. ViT processes each slice → embeddings
    2. Attention pooling combines slices → global features
    3. Classifier → lesion probability
    
    Supports independent dropout per plane for selective regularization.
    """
    def __init__(self, model_name="google/vit-base-patch16-224", num_classes=1, pretrained=True, 
                 pooling_mode='attention', dropout_input=0.3, dropout_dense=0.2):
        super().__init__()
        
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name) if pretrained else ViTModel(ViTModel.config_class())
        self.feature_dim = self.vit.config.hidden_size  # 768 for ViT-base
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with ViT
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
        
        # Reshape: (batch_size, num_slices, feature_dim)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class CNNMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with ResNet50 CNN.
    
    Architecture (IGUAL A ViT):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. ResNet50 backbone → features: (B*K, 2048)
    3. Reshape: (B, K, 2048)
    4. Attention pooling combines slices → global features: (B, 2048)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as ViT baseline).
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2):
        super().__init__()
        
        # Load pretrained ResNet50
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Feature extraction: everything except the last fc layer
        # ResNet50 outputs 2048-dim features
        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        self.feature_dim = 2048
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with ResNet50
        features = self.backbone(x)  # (B*K, 2048, 1, 1)
        features = features.view(batch_size * num_slices, self.feature_dim)  # (B*K, 2048)
        
        # Reshape: (B, K, 2048)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class ConvNeXtMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with ConvNeXt V2 Tiny CNN.
    
    Architecture (IDENTICAL TO ResNet50 pipeline):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. ConvNeXt V2 Tiny backbone → features: (B*K, 768)
    3. Reshape: (B, K, 768)
    4. Attention pooling combines slices → global features: (B, 768)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as ResNet50/ViT baseline).
    
    ConvNeXt V2 Tiny specs (with FCMAE pretrained weights):
    - 27.8M parameters (~same as ResNet50)
    - 768-dim feature space
    - Pre-trained on ImageNet-1K with FCMAE fine-tune
    - More efficient than ResNet50 with better feature quality
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2):
        super().__init__()
        
        # Load pretrained ConvNeXt V2 Tiny via timm
        # Using 'convnextv2_tiny.fcmae_ft_in1k' for pre-trained weights
        model_name = 'convnextv2_tiny.fcmae_ft_in1k' if pretrained else 'convnextv2_tiny'
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove classification head to get features only
        )
        
        # Feature extraction output dimension for ConvNeXt V2 Tiny:
        # ConvNeXt V2 Tiny outputs 768-dim features
        self.feature_dim = 768
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with ConvNeXt V2 Small
        features = self.backbone(x)  # (B*K, 768)
        
        # Reshape: (B, K, 768)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class SwinMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with Swin Transformer Tiny.
    
    Architecture (IDENTICAL TO ConvNeXt/ResNet50 pipeline):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. Swin Transformer Tiny backbone → features: (B*K, 768)
    3. Reshape: (B, K, 768)
    4. Attention pooling combines slices → global features: (B, 768)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as ResNet50/ViT/ConvNeXt baseline).
    
    Swin Transformer Tiny specs:
    - 28.3M parameters (~same as ResNet50 and ConvNeXt V2 Tiny)
    - 768-dim feature space
    - Pre-trained on ImageNet-1K
    - Window-based attention mechanism for efficiency
    - Better performance on medical imaging compared to standard CNNs
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2):
        super().__init__()
        
        # Load pretrained Swin Transformer Tiny via timm
        model_name = 'swin_tiny_patch4_window7_224.ms_in1k' if pretrained else 'swin_tiny_patch4_window7_224'
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove classification head to get features only
        )
        
        # Feature extraction output dimension for Swin Transformer Tiny:
        # Swin Tiny outputs 768-dim features (same as ConvNeXt V2 Tiny and ViT-base)
        self.feature_dim = 768
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with Swin Transformer Tiny
        features = self.backbone(x)  # (B*K, 768)
        
        # Reshape: (B, K, 768)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class SwinBaseMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with Swin Transformer Base.
    
    Architecture (IDENTICAL TO ConvNeXt/ResNet50/Swin Tiny pipeline):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. Swin Transformer Base backbone → features: (B*K, 1024)
    3. Reshape: (B, K, 1024)
    4. Attention pooling combines slices → global features: (B, 1024)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as ResNet50/ViT/ConvNeXt baseline).
    
    Swin Transformer Base specs:
    - 87.8M parameters (larger than Tiny 28.3M, better capacity for medical imaging)
    - 1024-dim feature space (vs 768-dim in Tiny)
    - Pre-trained on ImageNet-1K
    - Window-based attention mechanism for efficiency
    - Better performance on medical imaging compared to Swin Tiny
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2):
        super().__init__()
        
        # Load pretrained Swin Transformer Base via timm
        model_name = 'swin_base_patch4_window7_224.ms_in1k' if pretrained else 'swin_base_patch4_window7_224'
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # Remove classification head to get features only
        )
        
        # Feature extraction output dimension for Swin Transformer Base:
        # Swin Base outputs 1024-dim features (larger than Tiny's 768-dim)
        self.feature_dim = 1024
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with Swin Transformer Base
        features = self.backbone(x)  # (B*K, 1024)
        
        # Reshape: (B, K, 1024)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class ViTTinyMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with Vision Transformer Tiny from HuggingFace.
    
    Architecture (IDENTICAL TO ConvNeXt/ResNet50/Swin pipeline):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. ViT Tiny backbone → features: (B*K, 192)
    3. Reshape: (B, K, 192)
    4. Attention pooling combines slices → global features: (B, 192)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as other baselines).
    
    ViT Tiny specs (WinKawaks/vit-tiny-patch16-224):
    - ~5.7M parameters (mucho más pequeño que ResNet50/ConvNeXt/Swin)
    - 192-dim feature space
    - Pre-trained on ImageNet-1K
    - Patch-based attention mechanism
    - Ultra-fast inference with minimal parameters
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2, model_name="WinKawaks/vit-tiny-patch16-224"):
        super().__init__()
        
        # Load pretrained ViT Tiny from HuggingFace
        self.vit = ViTModel.from_pretrained(model_name) if pretrained else ViTModel(ViTModel.config_class())
        
        # Feature extraction output dimension for ViT Tiny:
        # ViT Tiny outputs 192-dim features (smaller than other models: 768 for base, 256 for small, 192 for tiny)
        self.feature_dim = self.vit.config.hidden_size  # 192
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        # Note: Adjusted to handle 192-dim input instead of 256/768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 128),  # Reduced from 256 due to smaller feature space
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with ViT Tiny
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token → (B*K, 192)
        
        # Reshape: (B, K, 192)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


class ViTSmallMultiSliceClassifier(nn.Module):
    """
    Classifier that processes multiple slices with Vision Transformer Small from HuggingFace.
    
    Architecture (IDENTICAL TO ConvNeXt/ResNet50/Swin/ViT-Tiny pipeline):
    1. Procesa K slices independientemente: (B, K, 3, 224, 224) → flatten → (B*K, 3, 224, 224)
    2. ViT Small backbone → features: (B*K, 768)
    3. Reshape: (B, K, 768)
    4. Attention pooling combines slices → global features: (B, 768)
    5. Classifier → lesion probability: (B, 1)
    
    Supports independent dropout per plane (same as other baselines).
    
    ViT Small specs (WinKawaks/vit-small-patch16-224):
    - ~22M parameters (intermediate between ViT-Tiny ~5.7M and ViT-Base ~86M)
    - 768-dim feature space (same as ConvNeXt/Swin/ViT-Base)
    - Pre-trained on ImageNet-1K
    - Patch-based attention mechanism
    - Good balance between efficiency and capacity
    """
    def __init__(self, num_classes=1, pretrained=True, pooling_mode='attention', 
                 dropout_input=0.3, dropout_dense=0.2, model_name="WinKawaks/vit-small-patch16-224"):
        super().__init__()
        
        # Load pretrained ViT Small from HuggingFace
        self.vit = ViTModel.from_pretrained(model_name) if pretrained else ViTModel(ViTModel.config_class())
        
        # Feature extraction output dimension for ViT Small:
        # ViT Small outputs 768-dim features (same as ConvNeXt/Swin/ViT-Base)
        self.feature_dim = self.vit.config.hidden_size  # 768
        
        # Pooling over slices
        self.pooling_mode = pooling_mode
        if pooling_mode == 'attention':
            self.pooling = AttentionPooling(self.feature_dim)
        elif pooling_mode == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling_mode == 'mean':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier with configurable dropout per plane
        # Architecture compatible with other baselines
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_input),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_dense),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: tensor (batch_size, num_slices, 3, 224, 224)
        Returns:
            logits: tensor (batch_size, 1)
            attention_weights: tensor (batch_size, num_slices) if pooling='attention'
        """
        batch_size, num_slices, C, H, W = x.shape
        
        # Flatten to process all slices at once
        # (B, K, 3, 224, 224) → (B*K, 3, 224, 224)
        x = x.view(batch_size * num_slices, C, H, W)
        
        # Extract features with ViT Small
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token → (B*K, 768)
        
        # Reshape: (B, K, 768)
        features = features.view(batch_size, num_slices, self.feature_dim)
        
        # Apply pooling
        attention_weights = None
        if self.pooling_mode == 'attention':
            pooled_features, attention_weights = self.pooling(features)
        elif self.pooling_mode == 'max':
            pooled_features = self.pooling(features.transpose(1, 2)).squeeze(-1)
        elif self.pooling_mode == 'mean':
            pooled_features = features.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled_features)
        
        if self.pooling_mode == 'attention':
            return logits, attention_weights
        else:
            return logits, None


# ==========================================
# LOSS FUNCTIONS
# ==========================================

class SmoothedBCELoss(nn.Module):
    """Binary cross-entropy loss with optional label smoothing and class weighting"""
    def __init__(self, smoothing=0.0, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        # pos_weight should be a tensor or scalar when used
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: tensor (batch_size,)
            targets: tensor (batch_size,) with values 0 or 1
        Returns:
            scalar loss
        """
        # Apply label smoothing
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Compute BCE loss with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weight,
            reduction='mean'
        )
        return loss


# ==========================================
# LEARNING RATE SCHEDULER
# ==========================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup + cosine annealing"""
    def __init__(self, optimizer, total_epochs, warmup_epochs, min_lr):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup (starts from epoch 1 with low LR)
            lr = self.optimizer.defaults['lr'] * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.optimizer.defaults['lr'] - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        """Return scheduler state"""
        return {'current_epoch': self.current_epoch}
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.current_epoch = state_dict.get('current_epoch', 0)

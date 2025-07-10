# models/elastic_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ElasticInteractionEnergyLoss(nn.Module):
    """
    Implementation of the Elastic Interaction Energy (EIE) Loss for lane detection.
    
    This loss treats lane predictions and ground truth as elastic curves and computes
    their interaction energy. It creates a long-range attractive force between predictions
    and ground truth, which helps with smooth and accurate lane detection.
    
    Based on the paper:
    "ElasticLaneNet: Exploiting Generalizable Human Keypoint Localization via Large Language Model"
    """
    def __init__(self, alpha=1.0, field_size=224, eps=1e-8):
        """
        Initialize the EIE loss.
        
        Args:
            alpha: Scaling factor for the prediction term (default: 1.0)
            field_size: Size of the field to represent lanes (default: 224)
            eps: Small value to avoid division by zero (default: 1e-8)
        """
        super(ElasticInteractionEnergyLoss, self).__init__()
        self.alpha = alpha
        self.field_size = field_size
        self.eps = eps
        
    def forward(self, pred_keypoints, gt_keypoints, valid_mask=None):
        """
        Compute the EIE loss between predicted and ground truth keypoints.
        
        Args:
            pred_keypoints: Tensor of shape [batch_size, num_lanes, num_points_per_lane, 2]
                          containing predicted keypoint coordinates normalized to [0,1]
            gt_keypoints: Tensor of shape [batch_size, num_lanes, num_points_per_lane, 2]
                        containing ground truth keypoint coordinates normalized to [0,1]
            valid_mask: Optional tensor of shape [batch_size, num_lanes, num_points_per_lane]
                      indicating which keypoints are valid (1) or invalid (0)
                      
        Returns:
            loss: EIE loss value (scalar tensor)
        """
        batch_size = pred_keypoints.shape[0]
        device = pred_keypoints.device
        loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            # Convert keypoints to fields
            gt_field = self.keypoints_to_field(gt_keypoints[b], valid_mask[b] if valid_mask is not None else None)
            pred_field = self.keypoints_to_field(pred_keypoints[b], valid_mask[b] if valid_mask is not None else None)
            
            # Compute difference field (Gt - αΨp) as in Equation (2)
            diff_field = gt_field - self.alpha * pred_field
            
            # Compute FFT of the difference field
            fft_diff = torch.fft.rfft2(diff_field)
            
            # Compute frequency magnitudes
            h, w = diff_field.shape
            freq_h = torch.fft.fftfreq(h, device=device)[:, None]
            freq_w = torch.fft.rfftfreq(w, device=device)[None, :]
            freq_magnitude = torch.sqrt(freq_h**2 + freq_w**2 + self.eps)
            
            # Apply equation (3) from the paper to compute gradient efficiently using FFT
            weighted_fft = freq_magnitude * fft_diff
            
            # Compute loss using inverse FFT (simplified approach)
            result_field = torch.fft.irfft2(weighted_fft)
            
            # Sum the squared values to get the energy
            batch_loss = torch.sum(result_field**2) / (h * w)
            loss += batch_loss
            
        return loss / batch_size
    
    def keypoints_to_field(self, keypoints, valid_mask=None):
        """
        Convert keypoints to a field representation.
        
        Args:
            keypoints: Tensor of shape [num_lanes, num_points_per_lane, 2]
                    containing normalized coordinates in range [0,1]
            valid_mask: Optional tensor of shape [num_lanes, num_points_per_lane]
                      indicating which keypoints are valid
                      
        Returns:
            field: Tensor of shape [field_size, field_size] representing the field
        """
        h, w = self.field_size, self.field_size
        field = torch.zeros((h, w), device=keypoints.device)
        
        num_lanes, num_points_per_lane = keypoints.shape[0], keypoints.shape[1]
        
        for lane_idx in range(num_lanes):
            # Skip lanes with no valid points
            if valid_mask is not None and not torch.any(valid_mask[lane_idx]):
                continue
                
            for i in range(num_points_per_lane - 1):
                p1 = keypoints[lane_idx, i]
                p2 = keypoints[lane_idx, i + 1]
                
                # Skip if any point is invalid
                if valid_mask is not None and (not valid_mask[lane_idx, i] or not valid_mask[lane_idx, i+1]):
                    continue
                
                # Skip if any coordinate is outside the normalized range [0,1]
                if torch.any(p1 < 0) or torch.any(p1 > 1) or torch.any(p2 < 0) or torch.any(p2 > 1):
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                p1_pixel = (int(p1[0].item() * (w-1)), int(p1[1].item() * (h-1)))
                p2_pixel = (int(p2[0].item() * (w-1)), int(p2[1].item() * (h-1)))
                
                # Draw line between consecutive points
                field = self.draw_line_on_field(field, p1_pixel, p2_pixel)
        
        return field
    
    def draw_line_on_field(self, field, p1, p2, value=1.0, width=2):
        """
        Draw a line on the field between two points using Bresenham's algorithm.
        
        Args:
            field: Tensor field to draw on
            p1: First point (x1, y1)
            p2: Second point (x2, y2)
            value: Value to set on the line
            width: Width of the line
            
        Returns:
            Updated field tensor
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # Use PyTorch's grid sampling for smoother lines
        h, w = field.shape
        
        # Create line parametrically for better quality
        num_steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
        num_steps = max(num_steps, 2)  # Ensure at least 2 steps
        
        t = torch.linspace(0, 1, steps=int(num_steps), device=field.device)
        line_x = x1 * (1 - t) + x2 * t
        line_y = y1 * (1 - t) + y2 * t
        
        # Convert to integer coordinates
        line_x = line_x.round().long()
        line_y = line_y.round().long()
        
        # Ensure coordinates are within bounds
        valid_mask = (line_x >= 0) & (line_x < w) & (line_y >= 0) & (line_y < h)
        line_x = line_x[valid_mask]
        line_y = line_y[valid_mask]
        
        # Draw line with specified width
        for i in range(-width//2, width//2 + 1):
            for j in range(-width//2, width//2 + 1):
                # Offset coordinates
                ox = line_x + i
                oy = line_y + j
                
                # Ensure offset coordinates are within bounds
                valid_offset = (ox >= 0) & (ox < w) & (oy >= 0) & (oy < h)
                ox = ox[valid_offset]
                oy = oy[valid_offset]
                
                # Set values at these coordinates
                if ox.numel() > 0:
                    field[oy, ox] = value
        
        return field


class FastElasticInteractionEnergyLoss(nn.Module):
    """
    A faster implementation of the Elastic Interaction Energy Loss using direct keypoint operations.
    
    Instead of creating fields and using FFT, this version directly computes the interaction
    energy between keypoints, which is more efficient for sparse keypoint representations.
    """
    def __init__(self, alpha=1.0, eps=1e-8):
        """
        Initialize the Fast EIE loss.
        
        Args:
            alpha: Scaling factor for the prediction term (default: 1.0)
            eps: Small value to avoid division by zero (default: 1e-8)
        """
        super(FastElasticInteractionEnergyLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        
    def forward(self, pred_keypoints, gt_keypoints, valid_mask=None):
        """
        Compute the EIE loss directly between keypoints.
        
        Args:
            pred_keypoints: Tensor of shape [batch_size, num_lanes, num_points_per_lane, 2]
            gt_keypoints: Tensor of shape [batch_size, num_lanes, num_points_per_lane, 2]
            valid_mask: Optional tensor of shape [batch_size, num_lanes, num_points_per_lane]
            
        Returns:
            loss: EIE loss value
        """
        batch_size = pred_keypoints.shape[0]
        num_lanes = pred_keypoints.shape[1]
        num_points = pred_keypoints.shape[2]
        device = pred_keypoints.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            batch_loss = torch.tensor(0.0, device=device)
            
            # For each lane, compute self-energy and interaction energy
            for lane_idx in range(num_lanes):
                # Skip lanes with no valid points
                if valid_mask is not None and not torch.any(valid_mask[b, lane_idx]):
                    continue
                    
                pred_lane = pred_keypoints[b, lane_idx]
                gt_lane = gt_keypoints[b, lane_idx]
                
                if valid_mask is not None:
                    lane_mask = valid_mask[b, lane_idx]
                    pred_lane = pred_lane[lane_mask]
                    gt_lane = gt_lane[lane_mask]
                
                if len(pred_lane) < 2 or len(gt_lane) < 2:
                    continue
                
                # Compute tangent vectors
                pred_tangents = self.compute_tangents(pred_lane)
                gt_tangents = self.compute_tangents(gt_lane)
                
                # Compute self-energy terms
                pred_self_energy = self.compute_self_energy(pred_lane, pred_tangents)
                gt_self_energy = self.compute_self_energy(gt_lane, gt_tangents)
                
                # Compute interaction energy between prediction and ground truth
                interaction_energy = self.compute_interaction_energy(
                    pred_lane, gt_lane, pred_tangents, gt_tangents
                )
                
                # Total energy (Eq. 1): Es + Ei (with opposite signs for attractive force)
                lane_loss = pred_self_energy + gt_self_energy - 2.0 * self.alpha * interaction_energy
                batch_loss += lane_loss
            
            total_loss += batch_loss
            
        return total_loss / batch_size
    
    def compute_tangents(self, points):
        """
        Compute tangent vectors for each segment of the lane.
        
        Args:
            points: Tensor of shape [num_points, 2] with keypoint coordinates
            
        Returns:
            tangents: Tensor of shape [num_points-1, 2] with normalized tangent vectors
        """
        # Calculate difference vectors between consecutive points
        diff = points[1:] - points[:-1]
        
        # Compute lengths of each segment
        lengths = torch.sqrt(torch.sum(diff**2, dim=1) + self.eps).unsqueeze(1)
        
        # Normalize to get unit tangent vectors
        tangents = diff / lengths
        
        return tangents
    
    def compute_self_energy(self, points, tangents):
        """
        Compute the self-energy of a lane.
        
        Args:
            points: Tensor of shape [num_points, 2] with keypoint coordinates
            tangents: Tensor of shape [num_points-1, 2] with tangent vectors
            
        Returns:
            energy: Self-energy value
        """
        num_segments = len(tangents)
        energy = torch.tensor(0.0, device=points.device)
        
        # Double loop over all segment pairs to compute interaction
        for i in range(num_segments):
            p1 = points[i]
            p2 = points[i+1]
            dl1 = tangents[i] * torch.norm(p2 - p1)
            
            # Self-interaction within the segment
            energy += torch.norm(dl1) / (8.0 * math.pi)
            
            # Interaction with other segments
            for j in range(i+1, num_segments):
                p3 = points[j]
                p4 = points[j+1]
                dl2 = tangents[j] * torch.norm(p4 - p3)
                
                # Compute distance between segment midpoints
                mid1 = (p1 + p2) / 2.0
                mid2 = (p3 + p4) / 2.0
                r = torch.norm(mid2 - mid1) + self.eps
                
                # Compute dot product of tangents
                dot_prod = torch.dot(dl1, dl2)
                
                # Add to energy
                energy += dot_prod / (8.0 * math.pi * r)
        
        return energy
    
    def compute_interaction_energy(self, pred_points, gt_points, pred_tangents, gt_tangents):
        """
        Compute the interaction energy between prediction and ground truth.
        
        Args:
            pred_points: Tensor of shape [num_pred_points, 2]
            gt_points: Tensor of shape [num_gt_points, 2]
            pred_tangents: Tensor of shape [num_pred_points-1, 2]
            gt_tangents: Tensor of shape [num_gt_points-1, 2]
            
        Returns:
            energy: Interaction energy value
        """
        num_pred_segments = len(pred_tangents)
        num_gt_segments = len(gt_tangents)
        energy = torch.tensor(0.0, device=pred_points.device)
        
        # Loop over all segment pairs to compute interaction
        for i in range(num_pred_segments):
            p1 = pred_points[i]
            p2 = pred_points[i+1]
            dl1 = pred_tangents[i] * torch.norm(p2 - p1)
            
            for j in range(num_gt_segments):
                p3 = gt_points[j]
                p4 = gt_points[j+1]
                dl2 = gt_tangents[j] * torch.norm(p4 - p3)
                
                # Compute distance between segment midpoints
                mid1 = (p1 + p2) / 2.0
                mid2 = (p3 + p4) / 2.0
                r = torch.norm(mid2 - mid1) + self.eps
                
                # Compute dot product of tangents (with negative sign for attraction)
                dot_prod = torch.dot(dl1, -dl2)  # Negative for attraction
                
                # Add to energy
                energy += dot_prod / (4.0 * math.pi * r)  # Factor of 1/4π instead of 1/8π
        
        return energy


def elastic_loss_scheduler(epoch, max_epochs, initial_weight=0.1, final_weight=1.0):
    """
    Schedule the weight of the elastic loss over training epochs.
    Gradually increasing the elastic loss weight can help stabilize training.
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        initial_weight: Initial weight for the elastic loss
        final_weight: Final weight for the elastic loss
        
    Returns:
        weight: Current weight for the elastic loss
    """
    progress = min(1.0, epoch / (0.7 * max_epochs))  # Reach final weight at 70% of training
    weight = initial_weight + progress * (final_weight - initial_weight)
    return weight
import argparse
from transformers import AutoTokenizer, AutoConfig
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from models import LocLLMModel
from datasets.culane import CULaneDataset
from dataclasses import dataclass

def transform_gt_to_padded_space(gt_lanes, original_width, original_height, padded_size=224):
    """
    Transform ground truth lanes from original image space to padded image space.
    This ensures GT and predictions are in the same coordinate system for visualization.
    
    Args:
        gt_lanes: List of lanes with normalized coordinates (relative to original image)
        original_width: Original image width (1640)
        original_height: Original image height (590)
        padded_size: Size of padded square image (224)
        
    Returns:
        Transformed lanes in padded space
    """
    # Calculate padding parameters (same as training)
    scale = padded_size / max(original_height, original_width)
    new_h, new_w = int(original_height * scale), int(original_width * scale)
    y_offset = (padded_size - new_h) // 2
    x_offset = (padded_size - new_w) // 2
    
    transformed_lanes = []
    for lane in gt_lanes:
        transformed_lane = []
        for x, y in lane:
            # Transform to padded space (same as training)
            x_padded = (x * new_w + x_offset) / padded_size
            y_padded = (y * new_h + y_offset) / padded_size
            
            # Clamp to valid range
            x_padded = max(0.0, min(1.0, x_padded))
            y_padded = max(0.0, min(1.0, y_padded))
            
            transformed_lane.append((x_padded, y_padded))
        transformed_lanes.append(transformed_lane)
    
    return transformed_lanes

def disable_torch_init():
    """Disable the redundant torch default initialization to accelerate model creation."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
PREFIX_IMAGE = "Image: "

@dataclass
class DataCollatorForLaneEvaluation(object):
    def __init__(self, image_token_len, conv_format):
        self.image_token_len = image_token_len
        self.conv_format = conv_format

    def __call__(self, instances):
        """Collate examples for lane evaluation using simplified approach."""
        batch_prompts = []
        batch_images = []
        batch_has_images = []
        result_dicts = []

        if self.conv_format == 'simple':
            from datasets.convsersation import conv_simple
            conv = conv_simple.copy()
        elif self.conv_format == 'keypoint':
            from datasets.convsersation import conv_keypoint
            conv = conv_keypoint.copy()
        else:
            from datasets.convsersation import conv_llama2
            conv = conv_llama2.copy()

        for i, instance in enumerate(instances):
            image = instance['image']
            image_path = instance['image_path']
            
            # Create prompts for each lane
            for lane_idx in range(4):  # Evaluate up to 4 lanes
                lane_name = f"lane{lane_idx+1}"
                position_names = ["leftmost", "center-left", "center-right", "rightmost"]
                position_name = position_names[lane_idx] if lane_idx < len(position_names) else f"lane{lane_idx+1}"
                
                lane_desc = f"The complete {position_name} lane on the road with all 10 points"
                lane_question = f"What are the coordinates of all 10 points along the {position_name} lane? Please list them in order from start to end."
                
                conv.messages = []
                if self.conv_format == 'keypoint':
                    conv.append_message(conv.roles[0], lane_desc)
                    conv.append_message(conv.roles[1], lane_question)
                    conv.append_message(conv.roles[2], None)
                elif self.conv_format == 'simple':
                    conv.append_message(conv.roles[0], lane_question)
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[0], lane_question)
                    conv.append_message(conv.roles[1], None)
                
                if self.conv_format == 'llama2':
                    conv.system = f"[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n".format(
                        system_message=PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
                    cur_prompt = conv.get_prompt()
                else:
                    text_inputs = conv.get_prompt()
                    cur_prompt = PREFIX_IMAGE + self.image_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs

                result_dict = {
                    'initial_prompt': cur_prompt,
                    'image_path': image_path,
                    'lane_idx': lane_idx,
                }
                
                batch_prompts.append(cur_prompt)
                batch_images.append(image)
                batch_has_images.append(True)
                result_dicts.append(result_dict)

        return result_dicts, batch_prompts, batch_images, batch_has_images

class CULaneValidationDataset(torch.utils.data.Dataset):
    """Simplified dataset for CULane validation with consistent preprocessing."""
    def __init__(self, list_file, image_folder, max_samples=None):
        self.image_paths = []
        self.image_folder = image_folder
        self.size = 224  # Model's input size
        
        # Image normalization (same as training)
        self.norm_mean = (0.48145466, 0.4578275, 0.40821073)
        self.norm_std = (0.26862954, 0.26130258, 0.27577711)
        
        # Read paths from the list file
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.image_paths.append(line)
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            
        print(f"Loaded {len(self.image_paths)} images for simplified validation")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        full_path = os.path.join(self.image_folder, image_path)
        
        # Load original image
        original_image = cv2.imread(full_path)
        if original_image is None:
            print(f"Warning: Could not read image {full_path}")
            original_image = np.zeros((590, 1640, 3), dtype=np.uint8)
        else:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = original_image.shape[:2]
        
        # Apply SAME preprocessing as training (resize with padding)
        processed_image = self._resize_with_padding(original_image, self.size)
        
        # Normalize image (same as training)
        image_tensor = torch.from_numpy(processed_image).float().permute(2, 0, 1) / 255.0
        for t, m, s in zip(image_tensor, self.norm_mean, self.norm_std):
            t.sub_(m).div_(s)
        
        print(f"DEBUG - Image {os.path.basename(image_path)}: {original_w}x{original_h} -> {self.size}x{self.size}")
        
        return {
            'image': image_tensor,
            'image_path': image_path,
            'original_image': original_image,
            'model_image': processed_image,
            'original_width': original_w,
            'original_height': original_h,
        }
    
    def _resize_with_padding(self, image, target_size):
        """
        Resize image to target_size x target_size while maintaining aspect ratio using padding.
        IDENTICAL to training preprocessing.
        """
        h, w = image.shape[:2]
        
        # Calculate scale to fit the larger dimension
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded

def parse_lane_points(output_text):
    """
    Parse lane points from model output text using simple approach.
    No coordinate swapping needed since preprocessing is consistent.
    
    Args:
        output_text: Text output from the model
        
    Returns:
        List of (x, y) normalized coordinates in [0,1] range
    """
    # Various regex patterns to handle different output formats
    coord_pattern1 = r"\(([0-9]*\.?[0-9]+),([0-9]*\.?[0-9]+)\)"  # (0.123,0.456)
    coord_pattern2 = r"\[([0-9]*\.?[0-9]+),([0-9]*\.?[0-9]+)\]"  # [0.123,0.456]
    coord_pattern3 = r"([0-9]*\.?[0-9]+),\s*([0-9]*\.?[0-9]+)"  # 0.123, 0.456
    
    print(f"DEBUG - Parsing lane points from: {output_text[:100]}...")
    
    points = []
    pattern_used = "none"
    
    # Try first pattern (x,y)
    matches = re.findall(coord_pattern1, output_text)
    if matches:
        pattern_used = "(x,y)"
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                
                # No coordinate swap needed - direct (x,y) format
                # Allow slightly out of range values but clip them
                if 0 <= x <= 1.1 and 0 <= y <= 1.1:
                    x = min(max(0.0, x), 1.0)
                    y = min(max(0.0, y), 1.0)
                    points.append((x, y))
                    
            except ValueError:
                continue
    
    # Try other patterns if first one fails
    if not points:
        matches = re.findall(coord_pattern2, output_text)
        if matches:
            pattern_used = "[x,y]"
            for match in matches:
                try:
                    x, y = float(match[0]), float(match[1])
                    if 0 <= x <= 1.1 and 0 <= y <= 1.1:
                        x = min(max(0.0, x), 1.0)
                        y = min(max(0.0, y), 1.0)
                        points.append((x, y))
                except ValueError:
                    continue
    
    if not points:
        matches = re.findall(coord_pattern3, output_text)
        if matches:
            pattern_used = "x, y"
            for match in matches:
                try:
                    x, y = float(match[0]), float(match[1])
                    if 0 <= x <= 1.1 and 0 <= y <= 1.1:
                        x = min(max(0.0, x), 1.0)
                        y = min(max(0.0, y), 1.0)
                        points.append((x, y))
                except ValueError:
                    continue
    
    print(f"DEBUG - Parsed {len(points)} points using pattern {pattern_used}")
    
    # Sort points by y-coordinate (top to bottom)
    if points:
        points.sort(key=lambda p: p[1])
        print(f"DEBUG - After sorting: {points[:3]} ... {points[-3:] if len(points) > 3 else points}")
    
    return points

def read_culane_gt(image_path, original_width, original_height):
    """
    Read CULane ground truth and normalize using same method as training.
    
    Args:
        image_path: Relative path to the image
        original_width: Original image width
        original_height: Original image height
        
    Returns:
        List of ground truth lanes with coordinates normalized to [0,1]
    """
    # Find the corresponding .lines.txt file
    ann_file = os.path.join("/storage/hay/culane_dataset", 
                           image_path.replace('.jpg', '.lines.txt'))
    
    if not os.path.exists(ann_file):
        print(f"GT file not found: {ann_file}")
        return []
    
    lanes = []
    try:
        with open(ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse coordinates
                values = line.split()
                points = []
                for i in range(0, len(values), 2):
                    if i+1 < len(values):
                        try:
                            x = float(values[i])
                            y = float(values[i+1])
                            if x >= 0 and y >= 0:
                                # SAME normalization as training: divide by original image dimensions
                                norm_x = x / original_width
                                norm_y = y / original_height
                                
                                # Clamp to [0,1] range
                                norm_x = max(0.0, min(1.0, norm_x))
                                norm_y = max(0.0, min(1.0, norm_y))
                                
                                points.append((norm_x, norm_y))
                        except ValueError:
                            continue
                
                if len(points) >= 2:
                    lanes.append(points)
        
        if lanes:
            print(f"Loaded {len(lanes)} GT lanes with points: {[len(lane) for lane in lanes]}")
    except Exception as e:
        print(f"Error reading GT file: {ann_file} - {e}")
    
    return lanes

def visualize_lanes(model_image, gt_lanes=None, pred_lanes=None, output_path="", 
                   original_width=None, original_height=None):
    """
    Visualize lanes on the model image (224x224) with proper coordinate alignment.
    
    Args:
        model_image: Processed model input image (224x224)
        gt_lanes: List of ground truth lanes (normalized coordinates)
        pred_lanes: List of predicted lanes (normalized coordinates)
        output_path: Path to save visualization
        original_width: Original image width for GT transformation
        original_height: Original image height for GT transformation
        
    Returns:
        Annotated image
    """
    model_size = 224
    img_copy = model_image.copy()
    
    # Transform GT lanes to padded space for fair comparison
    if gt_lanes and original_width and original_height:
        gt_lanes = transform_gt_to_padded_space(gt_lanes, original_width, original_height)
    
    # Colors
    gt_color = (0, 255, 0)    # Green for ground truth
    pred_color = (255, 0, 0)  # Red for predictions
    
    def draw_lane(image, lane_points, color, thickness=2, label=""):
        """Draw a single lane on the image."""
        if not lane_points or len(lane_points) < 2:
            return
            
        # Convert normalized coordinates to pixel coordinates
        pixel_points = []
        for norm_x, norm_y in lane_points:
            pixel_x = int(norm_x * model_size)
            pixel_y = int(norm_y * model_size)
            
            # Clamp to image bounds
            pixel_x = max(0, min(model_size-1, pixel_x))
            pixel_y = max(0, min(model_size-1, pixel_y))
            
            pixel_points.append((pixel_x, pixel_y))
        
        # Draw lane line
        pixel_points = np.array(pixel_points, dtype=np.int32)
        cv2.polylines(image, [pixel_points], False, color, thickness)
        
        # Draw start and end points
        if len(pixel_points) > 0:
            cv2.circle(image, tuple(pixel_points[0]), 3, color, -1)  # Start point
            cv2.circle(image, tuple(pixel_points[-1]), 3, color, -1)  # End point
            
            # Add label
            if label:
                cv2.putText(image, label, tuple(pixel_points[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw ground truth lanes
    if gt_lanes:
        for i, lane in enumerate(gt_lanes):
            draw_lane(img_copy, lane, gt_color, thickness=2, label=f"GT{i}")
    
    # Draw predicted lanes
    if pred_lanes:
        for i, lane in enumerate(pred_lanes):
            draw_lane(img_copy, lane, pred_color, thickness=2, label=f"P{i}")
    
    # Add legend
    legend_y = 20
    if gt_lanes:
        cv2.putText(img_copy, "GT (Green)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
        legend_y += 20
        
    if pred_lanes:
        cv2.putText(img_copy, "Pred (Red)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1)
    
    # Save visualization
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        print(f"Saved simple visualization: {output_path}")
    
    return img_copy

def calculate_metrics(pred_lanes, gt_lanes, iou_threshold=0.1, 
                     original_width=None, original_height=None):
    """
    Calculate metrics using simple IoU in normalized coordinate space.
    
    Args:
        pred_lanes: List of predicted lanes (normalized coordinates in padded space)
        gt_lanes: List of ground truth lanes (normalized coordinates in original space)
        iou_threshold: IoU threshold for considering a match
        original_width: Original image width for GT transformation
        original_height: Original image height for GT transformation
        
    Returns:
        (precision, recall, f1) metrics
    """
    model_size = 224
    
    # Transform GT lanes to padded space for fair comparison
    if gt_lanes and original_width and original_height:
        gt_lanes = transform_gt_to_padded_space(gt_lanes, original_width, original_height)
    
    # Create masks for each lane
    def create_lane_mask(lane_points):
        mask = np.zeros((model_size, model_size), dtype=np.uint8)
        if not lane_points or len(lane_points) < 2:
            return mask
            
        # Convert to pixel coordinates
        pixel_points = []
        for norm_x, norm_y in lane_points:
            pixel_x = int(norm_x * model_size)
            pixel_y = int(norm_y * model_size)
            pixel_x = max(0, min(model_size-1, pixel_x))
            pixel_y = max(0, min(model_size-1, pixel_y))
            pixel_points.append((pixel_x, pixel_y))
        
        if len(pixel_points) >= 2:
            pixel_points = np.array(pixel_points, dtype=np.int32)
            cv2.polylines(mask, [pixel_points], False, 1, thickness=3)
        
        return mask
    
    # Create masks
    pred_masks = [create_lane_mask(lane) for lane in pred_lanes]
    gt_masks = [create_lane_mask(lane) for lane in gt_lanes]
    
    num_pred = len(pred_masks)
    num_gt = len(gt_masks)
    
    if num_pred == 0 and num_gt == 0:
        return 1.0, 1.0, 1.0
    if num_pred == 0:
        return 0.0, 0.0, 0.0
    if num_gt == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate IoU matrix
    similarity_matrix = np.zeros((num_pred, num_gt))
    
    for i in range(num_pred):
        for j in range(num_gt):
            intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            iou = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = iou
    
    # Find best matches
    matches = []
    for i in range(num_pred):
        best_match = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i, best_match]
        if best_score >= iou_threshold:
            matches.append((i, best_match, best_score))
    
    # Remove duplicate matches (keep best scoring ones)
    final_matches = []
    matched_gt = set()
    for pred_idx, gt_idx, score in sorted(matches, key=lambda x: x[2], reverse=True):
        if gt_idx not in matched_gt:
            final_matches.append((pred_idx, gt_idx, score))
            matched_gt.add(gt_idx)
    
    # Calculate metrics
    tp = len(final_matches)
    fp = num_pred - tp
    fn = num_gt - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

@torch.no_grad()
def evaluate_culane(model, tokenizer, dataset, args):
    """
    Evaluate the model on CULane dataset using simplified consistent preprocessing.
    """
    image_token_len = model.config.num_patches
    debug = args.debug

    print(f"DEBUG - Starting simplified evaluation with image_token_len={image_token_len}")
    print(f"DEBUG - Debug mode: {'enabled' if debug else 'disabled'}")

    # Create data loader
    batch_size = args.batch_size
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=DataCollatorForLaneEvaluation(image_token_len, args.conv_format)
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_images = 0
    
    # Process batches
    for batch_idx, (result_dicts, batch_prompts, batch_images, batch_has_images) in enumerate(tqdm(data_loader, desc="Evaluating")):
        if batch_idx * batch_size >= args.max_samples:
            break
            
        if not result_dicts:
            continue
            
        print(f"\nDEBUG - Processing batch {batch_idx+1} with {len(batch_prompts)} prompts")
        
        # Get tokenized inputs
        tokenized_output = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        
        # Move tensors to GPU
        batch_images = torch.stack(batch_images).to(model.device)
        input_ids = torch.as_tensor(tokenized_output.input_ids).to(model.device)
        attention_mask = torch.as_tensor(tokenized_output.attention_mask).to(model.device)
        
        print(f"DEBUG - Input shape: {input_ids.shape}, Image shape: {batch_images.shape}")
        
        # Generate responses
        outputs = []
        for i in range(len(input_ids)):
            print(f"DEBUG - Generating response for prompt {i+1}/{len(input_ids)}")
            output = model.generate(
                input_ids[i:i+1],
                images=batch_images[i:i+1],
                has_images=[batch_has_images[i]],
                attention_mask=attention_mask[i:i+1],
                do_sample=False,
                max_new_tokens=200,
                num_beams=1,
            )
            outputs.append(output[0])
            
            if debug:
                input_token_len = input_ids[i].shape[0]
                output_text = tokenizer.decode(output[0][input_token_len:], skip_special_tokens=True).strip()
                print(f"DEBUG - Output: {output_text[:100]}...")
        
        # Group results by image
        image_results = {}
        for i, (result_dict, output_id) in enumerate(zip(result_dicts, outputs)):
            image_path = result_dict['image_path']
            lane_idx = result_dict['lane_idx']
            
            # Parse model output
            input_token_len = input_ids[i].shape[0]
            output_text = tokenizer.decode(output_id[input_token_len:], skip_special_tokens=True).strip()
            
            print(f"DEBUG - Processing result for {os.path.basename(image_path)}, lane {lane_idx}")
            
            if image_path not in image_results:
                image_results[image_path] = {
                    'lanes': [[] for _ in range(4)],  # Up to 4 lanes
                }
            
            # Parse lane points from output (no coordinate swapping needed)
            points = parse_lane_points(output_text)
            
            # Store lane points
            if points and len(points) >= 2:
                print(f"DEBUG - Storing {len(points)} points for lane {lane_idx}")
                image_results[image_path]['lanes'][lane_idx] = points
            else:
                print(f"DEBUG - No valid points found for lane {lane_idx}")
        
        # Process results for each image
        for image_path, results in image_results.items():
            print(f"\nDEBUG - Processing results for image: {os.path.basename(image_path)}")
            
            # Get dataset item for this image
            dataset_item = None
            for item in dataset:
                if item['image_path'] == image_path:
                    dataset_item = item
                    break
    
            if dataset_item is None:
                print(f"DEBUG - Dataset item not found: {image_path}")
                continue
            
            # Extract data from dataset item
            original_image = dataset_item['original_image']
            model_image = dataset_item['model_image']
            original_width = dataset_item['original_width']
            original_height = dataset_item['original_height']
            
            print(f"DEBUG - Original image: {original_width}x{original_height}")
            print(f"DEBUG - Model image: {model_image.shape}")
            
            # Filter empty lanes
            pred_lanes = [lane for lane in results['lanes'] if lane and len(lane) >= 2]
            print(f"DEBUG - Found {len(pred_lanes)} non-empty predicted lanes")
            
            # Print prediction ranges for debugging
            for lane_idx, lane in enumerate(pred_lanes):
                if lane:
                    x_values = [p[0] for p in lane]
                    y_values = [p[1] for p in lane]
                    print(f"DEBUG - Pred Lane {lane_idx}: X=[{min(x_values):.3f}-{max(x_values):.3f}], "
                          f"Y=[{min(y_values):.3f}-{max(y_values):.3f}]")
            
            # Read ground truth lanes with SAME normalization as training
            gt_lanes = read_culane_gt(image_path, original_width, original_height)
            if gt_lanes:
                print(f"DEBUG - Found {len(gt_lanes)} ground truth lanes")
                for i, lane in enumerate(gt_lanes):
                    if lane:
                        x_values = [p[0] for p in lane]
                        y_values = [p[1] for p in lane]
                        print(f"DEBUG - GT Lane {i}: X=[{min(x_values):.3f}-{max(x_values):.3f}], "
                              f"Y=[{min(y_values):.3f}-{max(y_values):.3f}]")
            else:
                print("DEBUG - No ground truth lanes found")
            
            # Check coordinate alignment (with transformed GT)
            if gt_lanes and pred_lanes:
                # Transform GT to padded space for comparison
                gt_lanes_transformed = transform_gt_to_padded_space(
                    gt_lanes, original_width, original_height
                )
                
                gt_x_range = [min([min([p[0] for p in lane]) for lane in gt_lanes_transformed]), 
                              max([max([p[0] for p in lane]) for lane in gt_lanes_transformed])]
                gt_y_range = [min([min([p[1] for p in lane]) for lane in gt_lanes_transformed]), 
                              max([max([p[1] for p in lane]) for lane in gt_lanes_transformed])]
                
                pred_x_range = [min([min([p[0] for p in lane]) for lane in pred_lanes]), 
                                max([max([p[0] for p in lane]) for lane in pred_lanes])]
                pred_y_range = [min([min([p[1] for p in lane]) for lane in pred_lanes]), 
                                max([max([p[1] for p in lane]) for lane in pred_lanes])]
                
                x_overlap = max(0, min(gt_x_range[1], pred_x_range[1]) - max(gt_x_range[0], pred_x_range[0]))
                y_overlap = max(0, min(gt_y_range[1], pred_y_range[1]) - max(gt_y_range[0], pred_y_range[0]))
                
                print(f"DEBUG - Coordinate overlap: X={x_overlap:.3f}, Y={y_overlap:.3f}")
                
                if x_overlap > 0.1 and y_overlap > 0.1:
                    print("✅ COORDINATE SPACES ARE ALIGNED!")
                else:
                    print("❌ Coordinate spaces still misaligned")

            # Create visualization  with coordinate transformation
            vis_path = os.path.join(args.output_dir, f"simple_{os.path.basename(image_path)}")
            visualized_img = visualize_lanes(
                model_image, gt_lanes, pred_lanes, vis_path,
                original_width=original_width, 
                original_height=original_height
            )
            
            # Calculate metrics
            # Calculate metrics with coordinate transformation
            if gt_lanes and pred_lanes:
                precision, recall, f1 = calculate_metrics(
                    pred_lanes, gt_lanes, 
                    iou_threshold=0.1,
                    original_width=original_width,
                    original_height=original_height
                )
                
                print(f"DEBUG - Metrics for {os.path.basename(image_path)}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_images += 1
            else:
                print(f"DEBUG - Skipping metrics: gt_lanes={len(gt_lanes) if gt_lanes else 0}, pred_lanes={len(pred_lanes)}")
    
    # Calculate final metrics
    if total_images > 0:
        avg_precision = total_precision / total_images
        avg_recall = total_recall / total_images
        avg_f1 = total_f1 / total_images
    else:
        avg_precision = avg_recall = avg_f1 = 0
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "simple_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'overall': {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'total_images': total_images
            },
            'preprocessing': 'simple_consistent'
        }, f, indent=2)
    
    # Print results
    print(f"\n\n=== SIMPLIFIED EVALUATION RESULTS ===")
    print(f"Evaluation completed on {total_images} images")
    print(f"Overall metrics: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}")
    print(f"Preprocessing: Simple consistent approach (no bbox/affine)")
    
    return avg_precision, avg_recall, avg_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--category", type=str, default="normal")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--conv-format", type=str, default="keypoint")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug", type=bool, default=True)
    args = parser.parse_args()

    # Set up environment
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Disable torch init
    disable_torch_init()
    
    # Load model and tokenizer
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left')
    model = LocLLMModel.from_pretrained(model_name, use_cache=True)
    
    # Convert model to bfloat16
    for name, param in model.model.named_parameters():
        if "lora_" not in name:
            param.data = param.data.bfloat16()
    model.lm_head.to(torch.bfloat16)
    model = model.to(device)
    
    # Create simplified dataset
    dataset = CULaneValidationDataset(
        list_file=args.question_file,
        image_folder=args.image_folder,
        max_samples=args.max_samples
    )
    
    # Evaluate model
    precision, recall, f1 = evaluate_culane(model, tokenizer, dataset, args)
    
    # Print results
    print(f"\nSimplified CULane Evaluation Results for category {args.category}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Preprocessing: Consistent simple approach")

if __name__ == "__main__":
    main()
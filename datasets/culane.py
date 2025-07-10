import transformers
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import logging
import random
from typing import Dict, List, Tuple
import os
import numpy as np
import cv2
import json
import math

from .constants import DEFAULT_IMAGE_PATCH_TOKEN, PREFIX_IMAGE, IGNORE_INDEX
from .convsersation import conv_simple, conv_llama2, conv_keypoint

# Define lane keypoint names - 10 keypoints per lane for up to 4 lanes
LANE_KEYPOINT_NAMES = []
for lane_id in range(1, 5):  # Assuming up to 4 lanes
    for point_idx in range(10):
        position = "start" if point_idx == 0 else "end" if point_idx == 9 else f"point{point_idx}"
        LANE_KEYPOINT_NAMES.append(f"lane{lane_id}_{position}")

# Enhanced lane descriptions with spatial anchoring
LANE_SPATIAL_DESCRIPTIONS = {
    'leftmost': "The leftmost lane marking that defines the left boundary of the drivable area, located closest to the LEFT edge of the image when looking from the driver's perspective",
    'center-left': "The center-left lane marking that separates the leftmost lane from the center portion of the road, positioned in the left-center area between the leftmost and center of the image",
    'center-right': "The center-right lane marking that separates the center portion of the road from the rightmost lane, positioned in the right-center area between the center and rightmost of the image",
    'rightmost': "The rightmost lane marking that defines the right boundary of the drivable area, located closest to the RIGHT edge of the image when looking from the driver's perspective",
    'center': "The center lane marking that divides the road approximately in half, typically running through the middle of the image",
    'left': "The left lane marking that defines the left traffic lane boundary, positioned on the left half of the image",
    'right': "The right lane marking that defines the right traffic lane boundary, positioned on the right half of the image",
    'single': "The single visible lane marking on the road surface, which may be a boundary or divider line"
}

# Spatial anchoring prompt template
SPATIAL_ANCHOR_PROMPT = """Looking at the road from the driver's perspective:
- Leftmost lane: The lane closest to the LEFT edge of the image
- Center-left lane: The lane between the leftmost and center
- Center-right lane: The lane between the center and rightmost  
- Rightmost lane: The lane closest to the RIGHT edge of the image
Remember: lanes are ordered from left to right as you look forward."""

# Enhanced structure descriptions with spatial context
ROAD_STRUCTURE_DESCRIPTIONS = {
    1: "a single-lane road configuration where one lane marking is visible, typically indicating a road edge or lane boundary",
    2: "a two-lane road configuration with left and right lane markings defining the drivable area boundaries, with the left lane on the LEFT side and right lane on the RIGHT side of the image",
    3: "a three-lane road configuration with distinct leftmost (LEFT edge), center (middle), and rightmost (RIGHT edge) lane markings creating two traffic lanes",
    4: "a multi-lane road configuration with four lane markings ordered left to right: leftmost (LEFT edge), center-left, center-right, and rightmost (RIGHT edge) defining three or more traffic lanes"
}

# Lane structure questions with spatial emphasis
LANE_STRUCTURE_QUESTIONS = [
    "What is the road structure and where are the lane markings located from left to right? Please describe their positions and provide coordinates.",
    "Describe the complete lane configuration in this image, listing lanes from LEFT to RIGHT including the number of lanes and their positions.",
    "How many lane markings are visible and what is their spatial arrangement from the LEFT edge to the RIGHT edge of the road?",
    "Provide a detailed description of the road structure and lane marking positions ordered from leftmost to rightmost."
]

# Individual lane questions with spatial context
LANE_COORDINATE_QUESTIONS = [
    "{spatial_context}\nWhere is the {position} lane in this image? Please provide its 10 coordinate points.",
    "{spatial_context}\nCould you identify the exact location of the {position} lane marking? List all 10 points along its trajectory.",
    "{spatial_context}\nWhat are the coordinates of the {position} lane marking from start to end?",
    "{spatial_context}\nPlease locate the {position} lane and provide its complete trajectory with all 10 coordinate points."
]

class CULaneDataset(Dataset):
    """
    CULane dataset with enhanced spatial understanding and consistent lane ordering.
    """
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict,
                 is_train=True
                 ):
        super().__init__()
        logging.warning("Loading CULane data with spatial anchoring...")
        self.size = 224
        self.num_joints = len(LANE_KEYPOINT_NAMES)  # 10 keypoints per lane * 4 lanes = 40
        
        # Convert CULane annotations if needed
        if data_path.endswith('.json'):
            # Already in processed format
            logging.info("Loading pre-processed annotations")
            with open(data_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            # Need to convert from CULane format
            logging.info("Converting CULane annotations")
            images_dir = multimodal_cfg['image_folder']
            self.annotations = self.preprocess_culane_annotations(data_path, images_dir)
            
            # Save processed annotations for future use
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            processed_path = data_path + '.processed_with_spatial.json'
            with open(processed_path, 'w') as f:
                json.dump(self.annotations, f)
            logging.info(f"Saved processed annotations to {processed_path}")
        
        list_data_dict = []
        instance_id = 0
        
        # Process annotations
        for image_info in self.annotations['images']:
            image_id = image_info['id']
            width = image_info['width']
            height = image_info['height']
            
            # Get annotations for this image
            lane_anns = [ann for ann in self.annotations['annotations'] 
                         if ann['image_id'] == image_id]
            
            # Group lanes by image instead of treating each lane as separate instance
            if lane_anns:
                # Combine all lanes for this image into one training instance
                all_lanes_keypoints = np.zeros((self.num_joints, 3), dtype=np.float32)
                all_lanes_keypoints_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
                
                # CRITICAL: Sort lanes by X-coordinate for consistent ordering
                lane_x_positions = []
                for lane_ann in lane_anns:
                    keypoints = np.array(lane_ann['keypoints']).reshape(-1, 3)
                    # Get average X position of visible points
                    visible_points = keypoints[keypoints[:, 2] > 0]
                    if len(visible_points) > 0:
                        avg_x = np.mean(visible_points[:, 0])
                        lane_x_positions.append((lane_ann, avg_x))
                
                # Sort lanes by X position (left to right)
                lane_x_positions.sort(key=lambda x: x[1])
                
                # Process lanes in sorted order
                for sorted_idx, (lane_ann, _) in enumerate(lane_x_positions):
                    if sorted_idx >= 4:  # Skip if more than 4 lanes
                        continue
                    
                    # Process lane keypoints
                    keypoints = np.array(lane_ann['keypoints']).reshape(-1, 3)
                    
                    # Use sorted index instead of original lane_id
                    start_idx = sorted_idx * 10
                    
                    for i in range(min(10, keypoints.shape[0])):
                        idx = start_idx + i
                        if idx < self.num_joints:
                            all_lanes_keypoints[idx, 0] = keypoints[i, 0]
                            all_lanes_keypoints[idx, 1] = keypoints[i, 1]
                            all_lanes_keypoints[idx, 2] = 0
                            
                            # Visibility flag
                            visibility = min(1, keypoints[i, 2])
                            all_lanes_keypoints_vis[idx, 0] = visibility
                            all_lanes_keypoints_vis[idx, 1] = visibility
                            all_lanes_keypoints_vis[idx, 2] = 0
                
                # Add this image instance to our dataset
                list_data_dict.append({
                    'file_name': image_info['file_name'],
                    'image_id': image_id,
                    'joints_3d': all_lanes_keypoints,
                    'joints_3d_vis': all_lanes_keypoints_vis,
                    'instance_id': instance_id,
                    'original_width': width,
                    'original_height': height
                })
                instance_id += 1
        
        logging.warning(f"Number of image instances: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg
        self.conv_format = self.multimodal_cfg.get("conv_format", "keypoint")
        
        # Set up conversation format
        if self.conv_format == 'simple':
            self.conv = conv_simple.copy()
        elif self.conv_format == 'keypoint':
            self.conv = conv_keypoint.copy()
        else:
            self.conv = conv_llama2.copy()
            
        print('Using Conversation Format:', self.conv_format)
        
        # Set up data augmentation
        if 'data_augmentation' in self.multimodal_cfg.keys():
            self.data_aug = self.multimodal_cfg['data_augmentation']
        else:
            self.data_aug = False
            
        # Image normalization
        if self.multimodal_cfg.get('dino_norm', False):
            norm_mean = (0.485, 0.456, 0.406)
            norm_std = (0.229, 0.224, 0.225)
        else:
            norm_mean = (0.48145466, 0.4578275, 0.40821073)
            norm_std = (0.26862954, 0.26130258, 0.27577711)
            
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        self.is_train = is_train
    
    def preprocess_culane_annotations(self, annotations_path: str, images_dir: str) -> dict:
        """
        Process the CULane dataset format with enhanced spatial awareness.
        """
        images = []
        annotations = []
        image_id = 0
        ann_id = 0
    
        # Get list of all image files
        image_files = []  
        if os.path.isdir(annotations_path):
            logging.info(f"Processing image directory: {annotations_path}")
            for root, _, files in os.walk(annotations_path):
                for file in files:
                    if file.endswith('.jpg'):
                        image_files.append(os.path.join(root, file))
        else:
            # It's a file listing all image files
            logging.info(f"Processing image list file: {annotations_path}")
            with open(annotations_path, 'r') as f:
                for line in f:
                    if line.strip():
                        image_files.append(line.strip())
    
        logging.info(f"Found {len(image_files)} image files")
    
        for img_path in image_files:
            # Get full image path
            full_img_path = os.path.join(images_dir, img_path)
            if not os.path.exists(full_img_path):
                logging.warning(f"Image not found: {full_img_path}")
                continue
        
            # Get corresponding annotation file
            ann_file = full_img_path.replace('.jpg', '.lines.txt')
            if not os.path.exists(ann_file):
                logging.warning(f"Annotation file not found: {ann_file}")
                continue
        
            # Create image entry
            img = cv2.imread(full_img_path)
            if img is None:
                logging.warning(f"Could not read image: {full_img_path}")
                continue
            
            h, w = img.shape[:2]
            images.append({
                'id': image_id,
                'file_name': img_path,
                'width': w,
                'height': h
            })
        
            # Process lane annotations with spatial ordering
            lane_annotations = []
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
                                    points.append((x, y))
                                except ValueError:
                                    continue
                    
                        if len(points) < 2:
                            continue
                    
                        # Calculate average X position for sorting
                        avg_x = np.mean([p[0] for p in points])
                        
                        lane_annotations.append({
                            'points': points,
                            'avg_x': avg_x
                        })
                
                # Sort lanes by X position (left to right)
                lane_annotations.sort(key=lambda x: x['avg_x'])
                
                # Create annotations with corrected lane IDs
                for lane_idx, lane_data in enumerate(lane_annotations):
                    if lane_idx >= 4:  # Maximum 4 lanes
                        break
                        
                    points = lane_data['points']
                    
                    # Resample to exactly 10 points
                    resampled_points = self.resample_lane(points, 10)
                    
                    # Create keypoints with normalization
                    keypoints = []
                    for x, y in resampled_points:
                        if 0 <= x <= w and 0 <= y <= h:
                            norm_x = x / w
                            norm_y = y / h
                            keypoints.extend([norm_x, norm_y, 2])  # 2 = visible
                        else:
                            keypoints.extend([0.0, 0.0, 0])  # 0 = invisible
                    
                    # Pad if needed
                    while len(keypoints) < 30:  # 10 points * 3 values each
                        keypoints.extend([0.0, 0.0, 0])
                    
                    # Create annotation with corrected lane_id based on position
                    annotations.append({
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': 1,
                        'keypoints': keypoints,
                        'num_keypoints': len(resampled_points),
                        'lane_id': lane_idx  # Now correctly ordered left to right
                    })
                    ann_id += 1
                    
            except Exception as e:
                logging.error(f"Error processing annotation file {ann_file}: {e}")
                continue
        
            image_id += 1
            if image_id % 100 == 0:
                logging.info(f"Processed {image_id} images with {ann_id} lane annotations")
    
        logging.info(f"Processed a total of {image_id} images with {ann_id} lane annotations")
        return {'images': images, 'annotations': annotations}
    
    def resample_lane(self, points: List[Tuple[float, float]], num_points: int = 10) -> List[Tuple[float, float]]:
        """
        Resample lane points to have exactly num_points points using linear interpolation.
        """
        if len(points) == 0:
            return []
            
        if len(points) == 1:
            return points * num_points
            
        # Calculate cumulative distances along the lane
        dists = [0]
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            dists.append(dists[-1] + dist)
        
        total_length = dists[-1]
        if total_length == 0:
            return [points[0]] * num_points
        
        # Interpolate points at equal distances
        resampled = []
        for i in range(num_points):
            target_dist = i * total_length / (num_points - 1) if num_points > 1 else 0
            
            # Find segment containing this distance
            idx = np.searchsorted(dists, target_dist) - 1
            idx = max(0, min(idx, len(points) - 2))
            
            # Interpolate between points idx and idx+1
            segment_length = dists[idx+1] - dists[idx]
            if segment_length > 0:
                alpha = (target_dist - dists[idx]) / segment_length
            else:
                alpha = 0
                
            x = points[idx][0] + alpha * (points[idx+1][0] - points[idx][0])
            y = points[idx][1] + alpha * (points[idx+1][1] - points[idx][1])
            
            resampled.append((x, y))
            
        return resampled
    
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        if self.is_train:
            while True:
                use_item, data_dict = self._parse_data_item(i)
                if use_item:
                    break
                else:
                    i = random.randint(0, self.__len__() - 1)
            return data_dict
        else:
            return self._parse_data_item_val(i)
    
    def _parse_data_item_val(self, i):
        """Process a single data item for validation."""
        sources = self.list_data_dict[i]
        result_dict = {}
        image, joints, joints_vis = self._get_simple_item(sources)
        image_id = sources['image_id']
        result_dict['images'] = image
        result_dict['image_id'] = image_id
        result_dict['joints'] = joints
        result_dict['joints_vis'] = joints_vis
        return result_dict
    
    def _parse_data_item(self, i) -> Tuple[bool, Dict[str, torch.Tensor]]:
        """
        Process a single data item for training with enhanced spatial understanding.
        """
        sources = self.list_data_dict[i]
        data_dict = {}
    
        image, joints, joints_vis = self._get_simple_item(sources)

        data_dict["image"] = image
        data_dict['has_image'] = True
        cur_token_len = 256
  
        # Get visible lanes - already sorted left to right due to preprocessing
        lanes = []
        
        for lane_idx in range(4):  # 4 possible lanes
            start_idx = lane_idx * 10
            
            # Check if at least 3 points in this lane are visible
            valid_points = 0
            
            for kpt_idx in range(10):
                idx = start_idx + kpt_idx
                if idx < self.num_joints:
                    x, y, v = joints[idx, 0], joints[idx, 1], joints_vis[idx, 0]
                    if v >= 1 and 0 <= x <= 1 and 0 <= y <= 1:
                        valid_points += 1
        
            if valid_points >= 3:
                lanes.append(lane_idx)
    
        # If no visible lanes, skip this item
        if not lanes:
            return False, {}
    
        # Lanes are already in left-to-right order
        # Assign structural positions based on count and position
        structural_positions = self._assign_structural_positions(lanes)
    
        kpt_name = []
        kpt_des = []
        question = []
        caption = []
    
        # Add structure-first question with spatial context (30% probability)
        if random.random() < 0.3:
            structure_prompt = random.choice(LANE_STRUCTURE_QUESTIONS)
            
            # Generate structure description with spatial emphasis
            num_lanes = len(lanes)
            
            # Use enhanced structure description
            structure_description = f"{SPATIAL_ANCHOR_PROMPT}\n\n{ROAD_STRUCTURE_DESCRIPTIONS[num_lanes]}"
            
            # Generate detailed caption with spatial context
            detailed_caption = self._generate_structure_caption_with_coordinates(
                lanes, structural_positions, joints, joints_vis
            )
            
            kpt_name.append("road_structure")
            kpt_des.append(structure_description)
            question.append(structure_prompt)
            caption.append(detailed_caption)
    
        # For each visible lane, include full description with spatial context
        for lane_id in lanes:
            structural_pos = structural_positions[lane_id]
            
            # Use enhanced spatial description
            spatial_description = LANE_SPATIAL_DESCRIPTIONS.get(
                structural_pos, 
                f"The {structural_pos} lane marking"
            )
        
            # Generate coordinate list for this lane
            lane_points_text = "["
            visible_points = 0
        
            for point_idx in range(10):
                idx = lane_id * 10 + point_idx
                if idx < self.num_joints:
                    x, y, v = joints[idx, 0], joints[idx, 1], joints_vis[idx, 0]
                    if v >= 1 and 0 <= x <= 1 and 0 <= y <= 1:
                        lane_points_text += f"({x:.3f},{y:.3f}),"
                        visible_points += 1
        
            # Consider a lane valid if at least 5 points are visible
            if visible_points >= 5:
                lane_points_text = lane_points_text.rstrip(',') + "]"
                
                # Create conversation with spatial context
                kpt_name.append(f"lane_{structural_pos}")
                kpt_des.append(spatial_description)
                
                # Use varied questions with spatial context
                question_template = random.choice(LANE_COORDINATE_QUESTIONS)
                question_text = question_template.format(
                    spatial_context=SPATIAL_ANCHOR_PROMPT,
                    position=structural_pos
                )
                question.append(question_text)
                
                caption.append(lane_points_text)
    
        if not kpt_name:
            return False, {}
    
        # Format conversation data
        self.conv.messages = []
        for idx in range(min(4, len(kpt_name))):
            if self.conv_format == 'keypoint':
                # LocLLM format: Description -> Question -> Answer
                self.conv.append_message(self.conv.roles[0], kpt_des[idx])    # Keypoint description
                self.conv.append_message(self.conv.roles[1], question[idx])   # Question
                self.conv.append_message(self.conv.roles[2], caption[idx])    # Answer (coordinates)
            elif self.conv_format == 'simple':
                self.conv.append_message(self.conv.roles[0], question[idx])
                self.conv.append_message(self.conv.roles[1], caption[idx])
            else:
                self.conv.append_message(self.conv.roles[0], question[idx])
                self.conv.append_message(self.conv.roles[1], caption[idx])
    
        # Format input text
        if self.conv_format == 'llama2':
            self.conv.system = f"[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n".format(
                system_message=PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN)
            text_inputs = self.conv.get_prompt()
        else:
            text_inputs = self.conv.get_prompt()
            text_inputs = PREFIX_IMAGE + cur_token_len * DEFAULT_IMAGE_PATCH_TOKEN + "\n" + text_inputs
      
        # Tokenize inputs
        inputs = self.tokenizer(text_inputs,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True).input_ids[0]
    
        # Prepare target labels
        target = inputs.clone()
        if self.conv_format == 'keypoint':
            sep = self.conv.sep1 + self.conv.roles[2] + ": "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
        elif self.conv_format == 'llama2':
            sep = self.conv.sep + self.conv.roles[1] + " "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids) + 2
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
        else:
            sep = self.conv.sep + self.conv.roles[1] + ": "
            rounds = text_inputs.split(self.conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break
                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(self.tokenizer(rou).input_ids)
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
                cur_len += round_len
    
        data_dict.update(dict(input_ids=inputs, labels=target))
        return True, data_dict
    
    def _assign_structural_positions(self, sorted_lanes):
        """
        Assign structural positions based on lane count and order.
        Lanes are already sorted left to right.
        """
        positions = {}
        num_lanes = len(sorted_lanes)
        
        if num_lanes == 1:
            positions[sorted_lanes[0]] = "single"
        elif num_lanes == 2:
            positions[sorted_lanes[0]] = "left"
            positions[sorted_lanes[1]] = "right"
        elif num_lanes == 3:
            positions[sorted_lanes[0]] = "leftmost"
            positions[sorted_lanes[1]] = "center"
            positions[sorted_lanes[2]] = "rightmost"
        elif num_lanes == 4:
            positions[sorted_lanes[0]] = "leftmost"
            positions[sorted_lanes[1]] = "center-left"
            positions[sorted_lanes[2]] = "center-right"
            positions[sorted_lanes[3]] = "rightmost"
        
        return positions
    
    def _generate_structure_caption_with_coordinates(self, sorted_lanes, positions, joints, joints_vis):
        """
        Generate structure caption with spatial context and lane trajectories.
        """
        num_lanes = len(sorted_lanes)
        
        # Start with lane count and spatial context
        if num_lanes == 1:
            caption = "There is 1 lane marking visible. "
        else:
            caption = f"There are {num_lanes} lane markings visible, ordered from left to right. "
        
        # Add detailed information for each lane with spatial emphasis
        for i, lane_id in enumerate(sorted_lanes):
            pos = positions[lane_id]
            start_idx = lane_id * 10
            
            # Get key points for trajectory description
            points = []
            for j in range(10):
                idx = start_idx + j
                if idx < self.num_joints and joints_vis[idx, 0] >= 1:
                    points.append((joints[idx, 0], joints[idx, 1]))
            
            if len(points) >= 3:
                # Get trajectory characteristics
                trajectory = self._analyze_lane_trajectory(lane_id, joints, joints_vis)
                
                # Add lane description with spatial context
                start_x, start_y = points[0]
                end_x, end_y = points[-1]
                
                caption += f"The {pos} lane (position {i+1} from left) {trajectory} from [{start_x:.3f},{start_y:.3f}] to [{end_x:.3f},{end_y:.3f}]. "
        
        return caption
    
    def _analyze_lane_trajectory(self, lane_id, joints, joints_vis):
        """
        Analyze lane trajectory with spatial awareness.
        """
        start_idx = lane_id * 10
        points = []
        
        for i in range(10):
            idx = start_idx + i
            if idx < self.num_joints and joints_vis[idx, 0] >= 1:
                points.append((joints[idx, 0], joints[idx, 1]))
        
        if len(points) < 3:
            return "extends forward"
        
        # Analyze horizontal deviation
        x_values = [p[0] for p in points]
        x_start, x_end = x_values[0], x_values[-1]
        x_change = x_end - x_start
        
        # Determine trajectory description with direction
        if abs(x_change) < 0.05:
            trajectory = "extends straight ahead"
        elif x_change > 0.1:
            trajectory = "curves toward the right"
        elif x_change < -0.1:
            trajectory = "curves toward the left"
        else:
            trajectory = "extends forward with slight deviation"
        
        return trajectory

    def _get_simple_item(self, sources):
        """
        Process image using simple resize with consistent preprocessing.
        """
        file_name = sources['file_name']
        image_folder = self.multimodal_cfg['image_folder']
        image_file = os.path.join(image_folder, file_name)
   
        # Load image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            logging.error(f"Could not read image: {image_file}")
            # Return a blank image and empty keypoints
            image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            return self.transforms(image), np.zeros((self.num_joints, 3)), np.zeros((self.num_joints, 3))
       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
  
        # Get keypoints (already normalized during preprocessing to original dimensions)
        joints = sources['joints_3d'].copy()
        joints_vis = sources['joints_3d_vis'].copy()
  
        # Calculate padding parameters BEFORE resizing
        scale = self.size / max(original_h, original_w)
        new_h, new_w = int(original_h * scale), int(original_w * scale)
        y_offset = (self.size - new_h) // 2
        x_offset = (self.size - new_w) // 2
    
        # Transform normalized coordinates to padded image space
        # Step 1: Convert from original normalized space to padded normalized space
        # For X: account for horizontal padding
        joints[:, 0] = (joints[:, 0] * new_w + x_offset) / self.size
  
        # For Y: account for vertical padding
        joints[:, 1] = (joints[:, 1] * new_h + y_offset) / self.size
  
        # Now resize image with padding
        image = self._resize_with_padding(image, self.size)
  
        # Apply data augmentation if enabled
        if self.data_aug and self.is_train:
            image, joints, joints_vis = self._apply_simple_augmentation(image, joints, joints_vis)
  
        # Apply image transforms (normalization)
        image = self.transforms(image)
  
        return image, joints, joints_vis
  
    def _resize_with_padding(self, image, target_size):
        """
        Resize image to target_size x target_size while maintaining aspect ratio using padding.
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
   
    def _apply_simple_augmentation(self, image, joints, joints_vis):
        """
        Apply simple data augmentation that maintains spatial relationships.
        """
        # Color augmentation (doesn't affect coordinates)
        if random.random() < 0.5:
            # Random brightness
            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
      
        if random.random() < 0.5:
            # Random contrast
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(image)
            image = np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
      
        # Horizontal flip (requires coordinate update and lane reordering)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # Horizontal flip
            joints, joints_vis = self._flip_joints_horizontal(joints, joints_vis)
      
        return image, joints, joints_vis
   
    def _flip_joints_horizontal(self, joints, joints_vis):
        """
        Flip joint coordinates horizontally and maintain left-right ordering.
        CRITICAL: After flipping, leftmost becomes rightmost, so we need to reorder.
        """
        joints = joints.copy()
        joints_vis = joints_vis.copy()
      
        # Flip X coordinates
        joints[:, 0] = 1.0 - joints[:, 0]
      
        # Reorder lanes to maintain left-to-right ordering after flip
        # Collect lanes
        lanes = []
        for lane_idx in range(4):
            start_idx = lane_idx * 10
            end_idx = start_idx + 10
            if end_idx <= self.num_joints:
                lane_joints = joints[start_idx:end_idx].copy()
                lane_vis = joints_vis[start_idx:end_idx].copy()
               
                # Check if lane has valid points
                if np.any(lane_vis[:, 0] > 0):
                    avg_x = np.mean(lane_joints[lane_vis[:, 0] > 0, 0])
                    lanes.append((avg_x, lane_joints, lane_vis))
       
        # Sort lanes by average X position (left to right)
        lanes.sort(key=lambda x: x[0])
       
        # Reassign lanes in order
        new_joints = joints.copy()
        new_joints_vis = joints_vis.copy()
       
        for new_idx, (_, lane_joints, lane_vis) in enumerate(lanes):
            start_idx = new_idx * 10
            end_idx = start_idx + 10
            new_joints[start_idx:end_idx] = lane_joints
            new_joints_vis[start_idx:end_idx] = lane_vis
       
        # Clear remaining lanes
        for idx in range(len(lanes), 4):
            start_idx = idx * 10
            end_idx = start_idx + 10
            new_joints[start_idx:end_idx] = 0
            new_joints_vis[start_idx:end_idx] = 0
      
        return new_joints, new_joints_vis
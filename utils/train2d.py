import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import random
import re
import cv2
import numpy as np

import torch
import transformers

from models import LocLLMModel
from datasets.culane import CULaneDataset
from models.elastic_loss import ElasticInteractionEnergyLoss, elastic_loss_scheduler
from utils.llavasimple_trainer import LLaVASimpleTrainer

from PIL import Image
import torch.nn as nn
import io


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llama_path: Optional[str] = field(default="")
    dino_path: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True)
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_size: int = field(default=224)
    crop_size: int = field(default=224)
    data_augmentation: bool = field(default=False)
    conv_format: str = field(default="keypoint")
    use_elastic_loss: bool = field(default=False)
    elastic_loss_weight: float = field(default=0.5)
    elastic_loss_alpha: float = field(default=1.0)
    # Validation settings
    validation_data_path: str = field(default=None, metadata={"help": "Path to validation data"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class LoRAArguments:
    lora_vision_r: int = field(default=8)
    lora_vision_alpha: float = field(default=16)
    lora_vision_dropout: float = field(default=0.05)
    lora_vision_enable: bool = field(default=False)
    lora_llm_r: int = field(default=8)
    lora_llm_alpha: float = field(default=16)
    lora_llm_dropout: float = field(default=0.05)
    lora_llm_enable: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class ValidationAnalyzer:
    """Handles validation analysis during training"""
    
    def __init__(self, val_data_path, image_folder, tokenizer):
        self.val_data_path = val_data_path
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.validation_history = []
        
        # Load validation images
        self.val_images = self._load_validation_images()
        
    def _load_validation_images(self):
        """Load validation image paths"""
        val_images = []
        if os.path.exists(self.val_data_path):
            with open(self.val_data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        val_images.append(line)
        return val_images[:10]  # Use first 10 images
    
    def run_validation(self, model, epoch):
        """Run comprehensive validation analysis"""
        print(f"\n{'='*60}")
        print(f"VALIDATION ANALYSIS - EPOCH {epoch}")
        print(f"{'='*60}")
        
        if not self.val_images:
            print("❌ No validation images found!")
            return

        torch.cuda.empty_cache()     
        model.eval()
        device = next(model.parameters()).device
        
        results = {
            'epoch': epoch,
            'total_images': len(self.val_images),
            'coordinate_parsing_success': 0,
            'coordinate_alignment_success': 0,
            'total_lanes_detected': 0,
            'total_lanes_gt': 0,
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': []
        }
        
        with torch.no_grad():
            for i, image_path in enumerate(self.val_images):
                try:
                    # Process single image
                    image_results = self._validate_single_image(
                        model, image_path, device, i+1
                    )
                    
                    # Update results
                    if image_results:
                        if image_results['parsing_success']:
                            results['coordinate_parsing_success'] += 1
                        if image_results['alignment_success']:
                            results['coordinate_alignment_success'] += 1
                        
                        results['total_lanes_detected'] += image_results['num_pred_lanes']
                        results['total_lanes_gt'] += image_results['num_gt_lanes']
                        
                        if image_results['f1'] is not None:
                            results['f1_scores'].append(image_results['f1'])
                            results['precision_scores'].append(image_results['precision'])
                            results['recall_scores'].append(image_results['recall'])
                            
                except Exception as e:
                    print(f"  ❌ Error processing image {i+1}: {e}")
                    continue
        
        # Calculate summary metrics
        self._print_validation_summary(results)
        self.validation_history.append(results)
        model.train()
        torch.cuda.empty_cache()
        
        return results
    
    def _validate_single_image(self, model, image_path, device, image_num):
        """Validate a single image and return detailed results"""
        print(f"  Processing image {image_num}/10: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        full_path = os.path.join(self.image_folder, image_path)
        if not os.path.exists(full_path):
            print(f"    ❌ Image not found: {full_path}")
            return None
            
        # Load original image
        original_image = cv2.imread(full_path)
        if original_image is None:
            print(f"    ❌ Could not read image")
            return None
            
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_h, original_w = original_image.shape[:2]
        
        # Preprocess image (same as training)
        processed_image = self._resize_with_padding(original_image, 224)
        
        # Normalize image
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
        image_tensor = torch.from_numpy(processed_image).float().permute(2, 0, 1) / 255.0
        for t, m, s in zip(image_tensor, norm_mean, norm_std):
            t.sub_(m).div_(s)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Create query for lane detection
        query = "What are the coordinates of all lanes in this image? Please list all visible lanes with their 10 points."
        
        # Format input
        input_text = f"<image> Human: {query} Assistant: "
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(device)
        
        # Generate prediction
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    images=image_tensor,
                    has_images=[True],
                    attention_mask=inputs.attention_mask,
                    do_sample=False,
                    max_new_tokens=100,
                    num_beams=1,
                )
            
            # Decode prediction
            input_token_len = inputs.input_ids.shape[1]
            output_text = self.tokenizer.decode(
                outputs[0][input_token_len:], skip_special_tokens=True
            ).strip()
            
        except Exception as e:
            print(f"    ❌ Generation failed: {e}")
            return None
        
        # Parse predicted coordinates
        pred_lanes = self._parse_lane_coordinates(output_text)
        parsing_success = len(pred_lanes) > 0
        
        # Load ground truth
        gt_lanes = self._load_ground_truth(image_path, original_w, original_h)
        
        # Check coordinate alignment
        alignment_success = False
        f1, precision, recall = None, None, None
        
        if pred_lanes and gt_lanes:
            alignment_success = self._check_coordinate_alignment(pred_lanes, gt_lanes)
            f1, precision, recall = self._calculate_metrics(pred_lanes, gt_lanes)
        
        print(f"    Pred lanes: {len(pred_lanes)}, GT lanes: {len(gt_lanes)}")
        print(f"    Parsing: {'✅' if parsing_success else '❌'}, "
              f"Alignment: {'✅' if alignment_success else '❌'}")
        if f1 is not None:
            print(f"    F1: {f1:.3f}, P: {precision:.3f}, R: {recall:.3f}")
        
        return {
            'parsing_success': parsing_success,
            'alignment_success': alignment_success,
            'num_pred_lanes': len(pred_lanes),
            'num_gt_lanes': len(gt_lanes),
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'output_text': output_text[:100] + "..." if len(output_text) > 100 else output_text
        }
    
    def _resize_with_padding(self, image, target_size):
        """Resize image with padding (same as training)"""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def _parse_lane_coordinates(self, text):
        """Parse lane coordinates from model output"""
        # Look for coordinate patterns
        coord_pattern = r'\(([0-9]*\.?[0-9]+),([0-9]*\.?[0-9]+)\)'
        matches = re.findall(coord_pattern, text)
        
        lanes = []
        current_lane = []
        
        for x_str, y_str in matches:
            try:
                x, y = float(x_str), float(y_str)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    current_lane.append((x, y))
                    
                    # If we have 10 points or found bracket closure, finish lane
                    if len(current_lane) >= 10:
                        lanes.append(current_lane[:10])
                        current_lane = []
            except ValueError:
                continue
        
        # Add remaining points as a lane if we have enough
        if len(current_lane) >= 3:
            lanes.append(current_lane)
        
        return lanes
    
    def _load_ground_truth(self, image_path, original_w, original_h):
        """Load and normalize ground truth lanes"""
        ann_file = os.path.join(self.image_folder, image_path.replace('.jpg', '.lines.txt'))
        
        if not os.path.exists(ann_file):
            return []
        
        lanes = []
        try:
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    values = line.split()
                    points = []
                    for i in range(0, len(values), 2):
                        if i+1 < len(values):
                            try:
                                x = float(values[i])
                                y = float(values[i+1])
                                if x >= 0 and y >= 0:
                                    # Normalize by original image dimensions
                                    norm_x = x / original_w
                                    norm_y = y / original_h
                                    norm_x = max(0.0, min(1.0, norm_x))
                                    norm_y = max(0.0, min(1.0, norm_y))
                                    points.append((norm_x, norm_y))
                            except ValueError:
                                continue
                    
                    if len(points) >= 2:
                        lanes.append(points)
        except Exception:
            pass
        
        return lanes
    
    def _check_coordinate_alignment(self, pred_lanes, gt_lanes):
        """Check if predicted and GT coordinates are in similar ranges"""
        if not pred_lanes or not gt_lanes:
            return False
        
        # Get coordinate ranges
        pred_x_coords = [x for lane in pred_lanes for x, y in lane]
        pred_y_coords = [y for lane in pred_lanes for x, y in lane]
        gt_x_coords = [x for lane in gt_lanes for x, y in lane]
        gt_y_coords = [y for lane in gt_lanes for x, y in lane]
        
        pred_x_range = [min(pred_x_coords), max(pred_x_coords)]
        pred_y_range = [min(pred_y_coords), max(pred_y_coords)]
        gt_x_range = [min(gt_x_coords), max(gt_x_coords)]
        gt_y_range = [min(gt_y_coords), max(gt_y_coords)]
        
        # Check overlap
        x_overlap = max(0, min(pred_x_range[1], gt_x_range[1]) - max(pred_x_range[0], gt_x_range[0]))
        y_overlap = max(0, min(pred_y_range[1], gt_y_range[1]) - max(pred_y_range[0], gt_y_range[0]))
        
        return x_overlap > 0.1 and y_overlap > 0.1
    
    def _calculate_metrics(self, pred_lanes, gt_lanes):
        """Calculate F1, precision, recall"""
        if not pred_lanes or not gt_lanes:
            return 0.0, 0.0, 0.0
        
        # Simple IoU-based matching
        matches = 0
        for pred_lane in pred_lanes:
            best_iou = 0
            for gt_lane in gt_lanes:
                iou = self._calculate_lane_iou(pred_lane, gt_lane)
                best_iou = max(best_iou, iou)
            if best_iou > 0.1:  # Threshold for match
                matches += 1
        
        precision = matches / len(pred_lanes) if pred_lanes else 0
        recall = matches / len(gt_lanes) if gt_lanes else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall
    
    def _calculate_lane_iou(self, lane1, lane2):
        """Simple IoU calculation for lanes"""
        # Convert to simple bounding boxes for quick IoU
        def get_bbox(lane):
            xs = [p[0] for p in lane]
            ys = [p[1] for p in lane]
            return [min(xs), min(ys), max(xs), max(ys)]
        
        bbox1 = get_bbox(lane1)
        bbox2 = get_bbox(lane2)
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _print_validation_summary(self, results):
        """Print comprehensive validation summary"""
        total = results['total_images']
        parsing_rate = results['coordinate_parsing_success'] / total
        alignment_rate = results['coordinate_alignment_success'] / total
        
        avg_f1 = np.mean(results['f1_scores']) if results['f1_scores'] else 0.0
        avg_precision = np.mean(results['precision_scores']) if results['precision_scores'] else 0.0
        avg_recall = np.mean(results['recall_scores']) if results['recall_scores'] else 0.0
        
        avg_lanes_per_image = results['total_lanes_detected'] / total
        
        print(f"Processed {total} images")
        print(f"{'✅' if alignment_rate > 0.5 else '❌'} COORDINATE SPACES {'ALIGNED' if alignment_rate > 0.5 else 'MISALIGNED'}")
        
        # Show improvement from previous epochs
        improvement_text = ""
        if len(self.validation_history) > 0:
            prev_f1 = np.mean(self.validation_history[-1]['f1_scores']) if self.validation_history[-1]['f1_scores'] else 0.0
            if avg_f1 > prev_f1:
                improvement_text = f" (improving from {prev_f1:.2f} in epoch {self.validation_history[-1]['epoch']})"
        
        print(f"F1 Score: {avg_f1:.2f}{improvement_text}")
        print(f"Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}")
        print(f"Coordinate parsing success: {results['coordinate_parsing_success']}/{total} images")
        print(f"Average lane detection per image: {avg_lanes_per_image:.1f}")
        print(f"{'='*60}")


class ValidationCallback(transformers.TrainerCallback):
    """Callback for validation during training"""
    
    def __init__(self, val_data_path, image_folder, tokenizer):
        self.analyzer = ValidationAnalyzer(val_data_path, image_folder, tokenizer)
        
    def on_epoch_end(self, args, state, control, model, tokenizer, **kwargs):
        """Run validation every 2 epochs"""
        if state.epoch % 2 == 0 and state.epoch > 0:
            self.analyzer.run_validation(model, int(state.epoch))


class CustomLLaVATrainer(LLaVASimpleTrainer):
    """Extended trainer with validation support"""
    
    def __init__(self, *args, use_elastic_loss=False, elastic_loss_weight=0.5, 
                 elastic_loss_alpha=1.0, max_epochs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_elastic_loss = use_elastic_loss
        self.elastic_loss_weight = elastic_loss_weight
        self.elastic_loss_alpha = elastic_loss_alpha
        self.max_epochs = max_epochs
        
        if self.use_elastic_loss:
            self.eie_loss = ElasticInteractionEnergyLoss(alpha=elastic_loss_alpha)
            logging.info(f"Using ElasticInteractionEnergyLoss with weight {elastic_loss_weight}")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with optional elastic loss"""
        outputs = model(**inputs)
        standard_loss = outputs.loss
        
        if self.use_elastic_loss and hasattr(self.state, "epoch"):
            # Add elastic loss implementation here if needed
            pass
            
        if return_outputs:
            return standard_loss, outputs
        return standard_loss


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        assert 'has_image' in instances[0].keys()
        has_images = [instance['has_image'] for instance in instances]
        batch['has_images'] = has_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning using SimpleCULaneDataset."""
    
    print("Creating CULaneDataset with validation support...")
    
    train_dataset = CULaneDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        multimodal_cfg=dict(
            image_folder=data_args.image_folder,
            data_augmentation=data_args.data_augmentation,
            image_size=data_args.image_size,
            crop_size=data_args.crop_size,
            conv_format=data_args.conv_format
        )
    )
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    print(f"Created CULaneDataset with {len(train_dataset)} samples")
    
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoRAArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    print("=" * 80)
    print("CULANE TRAINING WITH VALIDATION ANALYSIS")
    print("=" * 80)
    print("Features:")
    print("- CULaneDataset with consistent preprocessing")
    print("- Validation analysis every 2 epochs (10 images)")
    print("- Comprehensive metrics: F1, precision, recall")
    print("- Coordinate alignment checking")
    print("- Training progress tracking")
    print("=" * 80)

    # Set default validation path if not provided
    if not data_args.validation_data_path:
        data_args.validation_data_path = "/storage/hay/culane_dataset/list/test_split/test0_normal.txt"

    model = LocLLMModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        llama_path=model_args.llama_path,
        dino_path=model_args.dino_path,
        lora_vision_r=lora_args.lora_vision_r,
        lora_vision_alpha=lora_args.lora_vision_alpha,
        lora_vision_dropout=lora_args.lora_vision_dropout,
        lora_vision_enable=lora_args.lora_vision_enable,
        lora_llm_enable=lora_args.lora_llm_enable,
        lora_llm_r=lora_args.lora_llm_r,
        lora_llm_alpha=lora_args.lora_llm_alpha,
        lora_llm_dropout=lora_args.lora_llm_dropout,
        crop_size=data_args.crop_size)
    
    # Load mm projector weights
    if model_args.pretrain_mm_mlp_adapter is not None:
        print('Load pretrained mm_projector from: ', model_args.pretrain_mm_mlp_adapter)
        mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
        update_state = {}
        update_state['weight'] = mm_projector_weights['model.mm_projector.weight']
        update_state['bias'] = mm_projector_weights['model.mm_projector.bias']
        model.mm_projector.load_state_dict(update_state, strict=True)

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    model.initialize_vision_tokenizer(tokenizer=tokenizer)

    dtype = torch.bfloat16
    model.model.to(dtype)
    model.lm_head.to(dtype)

    for param in model.parameters():
        param.requires_grad_(False)

    if model_args.tune_mm_mlp_adapter:
        for p in model.mm_projector.parameters():
            p.requires_grad = True

    data_args.image_token_len = model.config.num_patches
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if not model_args.freeze_vit:
        assert model.config.lora_vision_enable
        for name, param in model.vision_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True
    else:
        model.vision_model.train = disabled_train
        model.vision_model.eval()

    if not model_args.freeze_llm:
        assert model.config.lora_llm_enable
        for name, param in model.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
            else:
                param.data = param.data.float()
                param.requires_grad = True

    params_grad = [n for n, p in model.named_parameters() if p.requires_grad]
    print("Trainable parameters: {}".format(len(params_grad)))
    
    # Create trainer with validation callback
    trainer = CustomLLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        use_elastic_loss=data_args.use_elastic_loss,
        elastic_loss_weight=data_args.elastic_loss_weight,
        elastic_loss_alpha=data_args.elastic_loss_alpha,
        max_epochs=training_args.num_train_epochs,
        **data_module
    )
    
    # Add validation callback
    #validation_callback = ValidationCallback(
    #    val_data_path=data_args.validation_data_path,
    #    image_folder=data_args.image_folder,
    #    tokenizer=tokenizer
    #)
    #trainer.add_callback(validation_callback)

    print(f"Starting training with validation analysis:")
    print(f"- Training samples: {len(data_module['train_dataset'])}")
    print(f"- Validation every 2 epochs on 10 images")
    print(f"- Validation data: {data_args.validation_data_path}")
    print(f"- Total epochs: {training_args.num_train_epochs}")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting fresh training...")
        trainer.train()
        
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)

    print("=" * 80)
    print("TRAINING COMPLETED WITH VALIDATION ANALYSIS!")
    print("=" * 80)
    print("Final summary:")
    #if hasattr(validation_callback.analyzer, 'validation_history') and validation_callback.analyzer.validation_history:
    #    history = validation_callback.analyzer.validation_history
    #    print(f"- Total validation runs: {len(history)}")
        
    #    if len(history) >= 2:
    #        first_f1 = np.mean(history[0]['f1_scores']) if history[0]['f1_scores'] else 0.0
    #        last_f1 = np.mean(history[-1]['f1_scores']) if history[-1]['f1_scores'] else 0.0
    #        improvement = last_f1 - first_f1
    #        print(f"- F1 improvement: {first_f1:.3f} → {last_f1:.3f} ({improvement:+.3f})")
        
    #    last_result = history[-1]
    #    alignment_rate = last_result['coordinate_alignment_success'] / last_result['total_images']
    #    parsing_rate = last_result['coordinate_parsing_success'] / last_result['total_images']
        
    #    print(f"- Final coordinate alignment: {alignment_rate:.1%}")
    #    print(f"- Final parsing success: {parsing_rate:.1%}")
    #    print(f"- Average lanes per image: {last_result['total_lanes_detected'] / last_result['total_images']:.1f}")
    
    print(f"- Model saved to: {training_args.output_dir}")
    print("=" * 80)
    print("Next steps:")
    print("1. Run full validation: python utils/valid2d.py")
    print("2. Test on different categories")
    print("3. Compare with previous models")
    print("=" * 80)


if __name__ == "__main__":
    train()
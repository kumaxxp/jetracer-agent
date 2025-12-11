"""ADE20K segmentation using OneFormer model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ADE20KSegmenter:
    """Wrapper for OneFormer ADE20K segmentation model."""

    def __init__(
        self,
        model_name: str = "shi-labs/oneformer_ade20k_swin_tiny",
        device: Optional[str] = None
    ):
        """
        Initialize ADE20K segmentation model.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        logger.info(f"Loading ADE20K segmentation model: {model_name}")
        logger.info("This may take a few minutes on first run...")

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load processor and model
        try:
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
            
            self.processor = OneFormerProcessor.from_pretrained(model_name)
            
            # Load model directly on target device to avoid meta tensor issues
            # For newer transformers versions, use device_map parameter
            try:
                # Try loading with device_map (newer transformers)
                self.model = OneFormerForUniversalSegmentation.from_pretrained(
                    model_name,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                logger.info("✓ Model loaded with device_map")
            except TypeError:
                # Fallback for older transformers without device_map support
                self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
                self.model = self.model.to(self.device)
                logger.info("✓ Model loaded with .to()")
            
            self.model.eval()
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Get ADE20K class names from model config
        config_id2label = self.model.config.id2label
        self.id2label = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in config_id2label.items()
        }

    def segment_image(self, image_path: str | Path) -> np.ndarray:
        """
        Segment an image and return ADE20K class IDs.

        Args:
            image_path: Path to input image

        Returns:
            Segmentation mask with ADE20K class IDs (H, W) as uint8
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size[::-1]  # (height, width)

        # Prepare inputs
        inputs = self.processor(
            images=image,
            task_inputs=["semantic"],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[original_size]
        )[0]

        # Convert to numpy
        ade20k_mask = predicted_semantic_map.cpu().numpy().astype(np.uint8)

        return ade20k_mask

    def segment_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Segment a numpy array image and return ADE20K class IDs.

        Args:
            image: Input image as numpy array (H, W, C) in RGB or BGR format

        Returns:
            Segmentation mask with ADE20K class IDs (H, W) as uint8
        """
        # Convert numpy to PIL
        if image.shape[2] == 3:
            # Assume BGR if coming from OpenCV
            pil_image = Image.fromarray(image[:, :, ::-1])
        else:
            pil_image = Image.fromarray(image)
        
        original_size = (image.shape[0], image.shape[1])  # (height, width)

        # Prepare inputs
        inputs = self.processor(
            images=pil_image,
            task_inputs=["semantic"],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[original_size]
        )[0]

        # Convert to numpy
        ade20k_mask = predicted_semantic_map.cpu().numpy().astype(np.uint8)

        return ade20k_mask

    def segment_batch(
        self,
        image_paths: list[str | Path],
        output_dir: str | Path,
        verbose: bool = True
    ) -> list[Path]:
        """
        Segment a batch of images and save results.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save segmentation masks
            verbose: Print progress

        Returns:
            List of output mask file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i, image_path in enumerate(image_paths):
            image_path = Path(image_path)

            if verbose:
                logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")

            try:
                # Segment image
                mask = self.segment_image(image_path)

                # Save mask as PNG
                output_path = output_dir / f"{image_path.stem}_seg.png"
                Image.fromarray(mask).save(output_path)
                output_paths.append(output_path)

                if verbose:
                    unique_classes = np.unique(mask)
                    logger.info(f"  Found {len(unique_classes)} classes")

            except Exception as e:
                logger.error(f"  Failed to process {image_path.name}: {e}")
                continue

        if verbose:
            logger.info(f"\n✓ Processed {len(output_paths)}/{len(image_paths)} images")

        return output_paths

    def get_class_name(self, class_id: int) -> str:
        """Get ADE20K class name from ID."""
        return self.id2label.get(class_id, f"unknown_{class_id}")

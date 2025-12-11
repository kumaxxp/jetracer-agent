"""TensorRT Segmentation for Jetson Orin Nano.

This module provides high-performance semantic segmentation using
NVIDIA TensorRT with FP16 quantization, optimized for Jetson devices.

Requirements (pre-installed on JetPack 6.2.1):
    - TensorRT
    - PyCUDA

Performance comparison (Jetson Orin Nano, 320x240 input):
    - OpenCV DNN CUDA FP16: ~25-40ms
    - TensorRT FP16: ~5-10ms
"""

import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

# TensorRT imports (available on Jetson with JetPack)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("[TensorRT] Not available, will fallback to OpenCV DNN")


class TensorRTSegmenter:
    """TensorRT-based semantic segmentation for Jetson.
    
    Features:
    - Automatic ONNX to TensorRT conversion with caching
    - FP16 quantization for optimal Jetson performance
    - GPU memory management with CUDA streams
    
    Example:
        segmenter = TensorRTSegmenter(
            onnx_path="/home/jetson/models/road_segmentation.onnx",
            input_size=(320, 240)
        )
        mask, time_ms = segmenter.inference(bgr_image)
    """
    
    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        onnx_path: str,
        input_size: Tuple[int, int] = (320, 240),
        fp16: bool = True,
        cache_dir: Optional[str] = None,
        max_workspace_size: int = 1 << 28,  # 256MB (Jetson-friendly)
        verbose: bool = False
    ):
        """Initialize TensorRT segmenter.
        
        Args:
            onnx_path: Path to ONNX model file
            input_size: (width, height) for model input
            fp16: Enable FP16 quantization (recommended)
            cache_dir: Directory to cache TensorRT engines
            max_workspace_size: Maximum GPU memory for TensorRT workspace
            verbose: Print detailed logs
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        self.onnx_path = Path(onnx_path)
        self.input_width, self.input_height = input_size
        self.fp16 = fp16
        self.cache_dir = Path(cache_dir) if cache_dir else self.onnx_path.parent
        self.max_workspace_size = max_workspace_size
        self.verbose = verbose
        
        # TensorRT components
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.stream = None
        
        # GPU buffers
        self.d_input = None
        self.d_output = None
        self.h_input = None
        self.h_output = None
        self.input_shape = None
        self.output_shape = None
        
        # Build or load engine
        self._load_engine()
        self._allocate_buffers()
        
        print(f"[TensorRT] Segmenter ready (FP16={fp16}, input={input_size})")
    
    def _get_engine_path(self) -> Path:
        """Get path for cached TensorRT engine."""
        suffix = "_fp16" if self.fp16 else "_fp32"
        return self.cache_dir / f"{self.onnx_path.stem}{suffix}.trt"
    
    def _load_engine(self):
        """Load TensorRT engine from cache or build from ONNX."""
        engine_path = self._get_engine_path()
        
        # Try to load cached engine
        if engine_path.exists():
            if self._is_cache_valid(engine_path):
                print(f"[TensorRT] Loading cached engine: {engine_path.name}")
                self._load_from_file(engine_path)
                return
            else:
                print("[TensorRT] Cached engine outdated, rebuilding...")
        
        # Build engine from ONNX
        print(f"[TensorRT] Building engine from: {self.onnx_path.name}")
        print(f"[TensorRT]   FP16={self.fp16}, input={self.input_width}x{self.input_height}")
        
        self._build_engine()
        self._save_engine(engine_path)
    
    def _is_cache_valid(self, engine_path: Path) -> bool:
        """Check if cached engine is still valid."""
        if self.onnx_path.exists():
            onnx_mtime = self.onnx_path.stat().st_mtime
            engine_mtime = engine_path.stat().st_mtime
            return engine_mtime > onnx_mtime
        return True
    
    def _build_engine(self):
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse failed: {errors}")
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
        
        # Enable FP16 if supported
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TensorRT]   FP16 mode enabled")
        elif self.fp16:
            print("[TensorRT]   Warning: FP16 not supported, using FP32")
        
        # Build engine (this takes a few minutes on first run)
        print("[TensorRT]   Building (this may take 1-3 minutes)...")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        elapsed = time.time() - start_time
        print(f"[TensorRT]   Engine built in {elapsed:.1f}s")
    
    def _save_engine(self, path: Path):
        """Save serialized engine to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.engine.serialize())
        print(f"[TensorRT]   Engine cached: {path.name}")
    
    def _load_from_file(self, path: Path):
        """Load engine from serialized file."""
        runtime = trt.Runtime(self.logger)
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from: {path}")
    
    def _allocate_buffers(self):
        """Allocate GPU memory buffers for inference."""
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Get tensor names (TensorRT 8.x API)
        input_name = self.engine.get_tensor_name(0)
        output_name = self.engine.get_tensor_name(1)
        
        # Get shapes
        self.input_shape = self.engine.get_tensor_shape(input_name)
        self.output_shape = self.engine.get_tensor_shape(output_name)
        
        if self.verbose:
            print(f"[TensorRT] Input shape: {self.input_shape}")
            print(f"[TensorRT] Output shape: {self.output_shape}")
        
        # Calculate sizes
        input_size = int(np.prod(self.input_shape)) * 4  # float32 = 4 bytes
        output_size = int(np.prod(self.output_shape)) * 4
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        
        # Allocate page-locked host memory for faster transfers
        self.h_input = cuda.pagelocked_empty(
            int(np.prod(self.input_shape)), dtype=np.float32
        )
        self.h_output = cuda.pagelocked_empty(
            int(np.prod(self.output_shape)), dtype=np.float32
        )
        
        # Set tensor addresses for execution
        self.context.set_tensor_address(input_name, int(self.d_input))
        self.context.set_tensor_address(output_name, int(self.d_output))
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess BGR image for TensorRT inference.
        
        Args:
            image: BGR image (H, W, 3), uint8
            
        Returns:
            Preprocessed tensor as contiguous float32 array
        """
        # Resize
        resized = cv2.resize(
            image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        
        # HWC to NCHW format
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return np.ascontiguousarray(tensor, dtype=np.float32)
    
    def inference(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference on BGR image.
        
        Args:
            image: BGR image (H, W, 3), uint8
            
        Returns:
            Tuple of:
                - mask: Segmentation mask (H, W), uint8 (resized to original)
                - inference_time_ms: Inference time in milliseconds
        """
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        np.copyto(self.h_input, input_tensor.ravel())
        
        # Copy to GPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        
        # Execute inference
        start_time = time.perf_counter()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Copy from GPU
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        
        # Reshape and postprocess
        output = self.h_output.reshape(self.output_shape)
        
        # Remove batch dimension if present
        if output.ndim == 4:
            output = output[0]  # (C, H, W)
        
        # Argmax over classes
        mask = np.argmax(output, axis=0).astype(np.uint8)
        
        # Resize to original size
        mask = cv2.resize(
            mask,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        return mask, inference_time
    
    def warmup(self, n_iterations: int = 5):
        """Warmup the engine with dummy inputs."""
        dummy = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        
        print(f"[TensorRT] Warming up ({n_iterations} iterations)...")
        times = []
        for _ in range(n_iterations):
            _, t = self.inference(dummy)
            times.append(t)
        
        avg_time = np.mean(times[1:])  # Skip first
        print(f"[TensorRT] Warmup complete, avg: {avg_time:.2f}ms")
    
    def __del__(self):
        """Cleanup GPU resources."""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.stream.synchronize()
            except:
                pass


class SegmentationModel:
    """Segmentation model with TensorRT backend and OpenCV DNN fallback.
    
    This class automatically uses TensorRT if available, otherwise falls
    back to OpenCV DNN for compatibility.
    """
    
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (320, 240)):
        self.model_path = model_path
        self.input_size = input_size
        self.backend = None  # "tensorrt" or "opencv"
        self.segmenter = None
        self.net = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model with TensorRT or fallback to OpenCV DNN."""
        if not Path(self.model_path).exists():
            print(f"[Segmentation] Model not found: {self.model_path}")
            return
        
        # Try TensorRT first
        if TENSORRT_AVAILABLE:
            try:
                self.segmenter = TensorRTSegmenter(
                    onnx_path=self.model_path,
                    input_size=self.input_size,
                    fp16=True
                )
                self.segmenter.warmup(3)
                self.backend = "tensorrt"
                print("[Segmentation] Using TensorRT backend")
                return
            except Exception as e:
                print(f"[Segmentation] TensorRT failed: {e}")
                print("[Segmentation] Falling back to OpenCV DNN")
        
        # Fallback to OpenCV DNN
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                print("[Segmentation] Using OpenCV DNN (CUDA FP16)")
            except Exception:
                print("[Segmentation] Using OpenCV DNN (CPU)")
            self.backend = "opencv"
        except Exception as e:
            print(f"[Segmentation] Failed to load model: {e}")
    
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run segmentation analysis on frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Dict with road_ratio, road_center_x, inference_time_ms, available
        """
        if frame is None:
            return self._empty_result()
        
        if self.backend == "tensorrt":
            return self._analyze_tensorrt(frame)
        elif self.backend == "opencv":
            return self._analyze_opencv(frame)
        else:
            return self._empty_result()
    
    def _analyze_tensorrt(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze using TensorRT backend."""
        try:
            mask, time_ms = self.segmenter.inference(frame)
            road_mask = (mask == 1)  # 1 = ROAD
            
            road_ratio = road_mask.sum() / road_mask.size
            
            if road_mask.any():
                road_center_x = np.where(road_mask)[1].mean() / road_mask.shape[1]
            else:
                road_center_x = 0.5
            
            return {
                "road_ratio": round(float(road_ratio), 3),
                "road_center_x": round(float(road_center_x), 3),
                "inference_time_ms": round(time_ms, 2),
                "backend": "tensorrt",
                "available": True
            }
        except Exception as e:
            print(f"[Segmentation] TensorRT error: {e}")
            return self._empty_result()
    
    def _analyze_opencv(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze using OpenCV DNN backend."""
        try:
            start_time = time.perf_counter()
            
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, self.input_size, swapRB=True
            )
            self.net.setInput(blob)
            output = self.net.forward()
            
            time_ms = (time.perf_counter() - start_time) * 1000
            
            mask = np.argmax(output[0], axis=0)
            road_mask = (mask == 1)
            
            road_ratio = road_mask.sum() / road_mask.size
            
            if road_mask.any():
                road_center_x = np.where(road_mask)[1].mean() / road_mask.shape[1]
            else:
                road_center_x = 0.5
            
            return {
                "road_ratio": round(float(road_ratio), 3),
                "road_center_x": round(float(road_center_x), 3),
                "inference_time_ms": round(time_ms, 2),
                "backend": "opencv",
                "available": True
            }
        except Exception as e:
            print(f"[Segmentation] OpenCV error: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result when segmentation unavailable."""
        return {
            "road_ratio": 0.0,
            "road_center_x": 0.5,
            "inference_time_ms": 0.0,
            "backend": None,
            "available": False
        }
    
    def get_mask(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Get raw segmentation mask.
        
        Args:
            frame: BGR image
            
        Returns:
            Tuple of (mask, inference_time_ms) or (None, 0) if unavailable
        """
        if frame is None:
            return None, 0.0
        
        if self.backend == "tensorrt":
            return self.segmenter.inference(frame)
        elif self.backend == "opencv":
            try:
                start_time = time.perf_counter()
                blob = cv2.dnn.blobFromImage(
                    frame, 1/255.0, self.input_size, swapRB=True
                )
                self.net.setInput(blob)
                output = self.net.forward()
                time_ms = (time.perf_counter() - start_time) * 1000
                
                mask = np.argmax(output[0], axis=0).astype(np.uint8)
                # Resize to original frame size
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
                return mask, time_ms
            except Exception as e:
                print(f"[Segmentation] Error: {e}")
                return None, 0.0
        
        return None, 0.0
    
    def is_ready(self) -> bool:
        """Check if model is ready."""
        return self.backend is not None
    
    def get_backend(self) -> Optional[str]:
        """Get current backend name."""
        return self.backend

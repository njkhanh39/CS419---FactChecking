import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLIModel:
    """
    Wrapper for RoBERTa-MNLI model to perform Natural Language Inference.
    Determines if a premise (evidence) supports, refutes, or is neutral to a hypothesis (claim).
    """

    # UPDATED: Use the official repository ID. 
    # Recommended models (best to fastest):
    # 1. "MoritzLaurer/deberta-v3-large-zeroshot-v2.0" - Best accuracy, ~5-8s (RECOMMENDED)
    # 2. "microsoft/deberta-v3-base" - Good balance, ~2-3s
    # 3. "FacebookAI/roberta-large-mnli" - Good accuracy, ~10s
    # 4. "typeform/distilroberta-base-v2" - Fast but lower accuracy, ~1s
    # 5. "cross-encoder/nli-distilroberta-base" - Very fast but poor accuracy, avoid
    def __init__(self, model_name="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", device=None, use_quantization=False, use_onnx=False):
        """
        Initialize the NLI model.
        
        Args:
            model_name (str): HuggingFace model name (default: MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
            device (str): 'cuda', 'mps', or 'cpu'. If None, auto-detects.
            use_quantization (bool): Use int8 quantization for faster inference (reduces memory by ~75%)
            use_onnx (bool): Use ONNX Runtime for optimized CPU inference (requires onnxruntime)
        """
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_onnx = use_onnx
        
        # Auto-detect device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Loading NLI model: {model_name} on {self.device}...")
        if use_quantization:
            logger.info("  ⚡ Using INT8 quantization for faster inference")
        if use_onnx:
            logger.info("  ⚡ Using ONNX Runtime optimization")
        
        # Log GPU info if available
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with optimizations
            if use_onnx:
                # ONNX works on both CPU and GPU
                self._load_onnx_model(model_name)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Apply quantization if requested (CPU only)
                if use_quantization and self.device.type == "cpu":
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("  ✓ INT8 quantization applied (4x smaller, 2-3x faster)")
                
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode (PyTorch only)
            
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise e

        # Detect label mapping based on model
        # DeBERTa models: 0 = Entailment, 1 = Neutral, 2 = Contradiction
        # RoBERTa models: 0 = Contradiction, 1 = Neutral, 2 = Entailment
        if "deberta" in model_name.lower():
            self.label_map = {0: "SUPPORT", 1: "NEUTRAL", 2: "REFUTE"}
            logger.info("Using DeBERTa label mapping: 0=SUPPORT, 1=NEUTRAL, 2=REFUTE")
        else:
            self.label_map = {0: "REFUTE", 1: "NEUTRAL", 2: "SUPPORT"}
            logger.info("Using RoBERTa label mapping: 0=REFUTE, 1=NEUTRAL, 2=SUPPORT")
    
    def _load_onnx_model(self, model_name: str):
        """Load ONNX-optimized model (requires onnxruntime and optimum)"""
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            
            logger.info("Loading ONNX-optimized model...")
            
            # Choose provider based on device
            if self.device.type == "cuda":
                provider = "CUDAExecutionProvider"
                logger.info("  🎮 Using ONNX with CUDA GPU acceleration")
                # For GPU, install: pip install onnxruntime-gpu optimum[onnxruntime-gpu]
            else:
                provider = "CPUExecutionProvider"
                logger.info("  💻 Using ONNX with CPU optimization")
            
            # Check if ONNX quantization is requested (CPU only)
            onnx_quantize = os.getenv("ONNX_QUANTIZE", "false").lower() == "true"
            
            # Define cache directories
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_name = model_name.replace("/", "--")
            onnx_cache_path = os.path.join(cache_dir, f"models--{model_cache_name}")
            quantized_cache_path = "./onnx_quantized"
            
            if onnx_quantize and self.device.type == "cpu" and self.use_quantization:
                # Check if quantized model already exists
                if os.path.exists(quantized_cache_path) and os.path.exists(os.path.join(quantized_cache_path, "model_quantized.onnx")):
                    logger.info("  📦 Loading cached ONNX INT8 quantized model...")
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        quantized_cache_path,
                        provider=provider
                    )
                    logger.info("  ✓ ONNX INT8 quantized model loaded from cache (instant!)")
                else:
                    logger.info("  ⚡ Creating ONNX INT8 quantized model (first time only)...")
                    # Load and quantize
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        model_name,
                        export=True,
                        provider=provider
                    )
                    
                    # Apply dynamic quantization
                    quantizer = ORTQuantizer.from_pretrained(self.model)
                    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
                    quantizer.quantize(save_dir=quantized_cache_path, quantization_config=qconfig)
                    
                    # Reload quantized model
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        quantized_cache_path,
                        provider=provider
                    )
                    logger.info("  ✓ ONNX INT8 quantized model created and saved (4x smaller, 3-5x faster)")
            else:
                # Check if ONNX model already exists in cache
                # ONNX models are saved with "onnx" subfolder in the cache
                onnx_exists = any([
                    os.path.exists(os.path.join(onnx_cache_path, "onnx", "model.onnx")),
                    os.path.exists(os.path.join(onnx_cache_path, "snapshots"))  # Check snapshots folder
                ])
                
                if onnx_exists:
                    logger.info("  📦 Loading cached ONNX model...")
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        model_name,
                        export=False,  # Don't re-export, use cached version
                        provider=provider
                    )
                    logger.info("  ✓ ONNX model loaded from cache (instant!)")
                else:
                    logger.info("  🔄 Converting to ONNX format (first time only, ~2min)...")
                    self.model = ORTModelForSequenceClassification.from_pretrained(
                        model_name,
                        export=True,  # Convert to ONNX first time
                        provider=provider
                    )
                    logger.info("  ✓ ONNX model created and cached (2-5x faster)")
            
            # ONNX models don't need .eval() - they're always in inference mode
        except ImportError as e:
            if self.device.type == "cuda":
                logger.warning("onnxruntime-gpu not installed. Install with: pip install onnxruntime-gpu optimum[onnxruntime-gpu]")
            else:
                logger.warning("optimum/onnxruntime not installed. Install with: pip install optimum[onnxruntime]")
            logger.info("Falling back to standard PyTorch model...")
            self.use_onnx = False
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"ONNX loading failed: {e}")
            logger.info("Falling back to standard PyTorch model...")
            self.use_onnx = False
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

    def predict_batch(self, premises: list[str], hypothesis: str) -> list[dict]:
        """
        Run inference on a batch of premise sentences against a single hypothesis (claim).
        
        Args:
            premises (list[str]): List of evidence sentences.
            hypothesis (str): The user claim.
            
        Returns:
            list[dict]: List of results with label and probabilities.
        """
        if not premises:
            return []

        # Create pairs: [ (premise_1, hypothesis), (premise_2, hypothesis), ... ]
        # Note: RoBERTa expects (premise, hypothesis) usually joined by separator
        pairs = [[premise, hypothesis] for premise in premises]

        # Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        # Format results
        results = []
        cpu_probs = probs.cpu().numpy()

        for i, prob_dist in enumerate(cpu_probs):
            # Validate output dimensions
            num_labels = len(prob_dist)
            if num_labels != 3:
                logger.error(f"Model returned {num_labels} labels instead of 3! ONNX export may be broken.")
                logger.error(f"Probabilities: {prob_dist}")
                raise ValueError(
                    f"Model output has {num_labels} labels (expected 3). "
                    f"ONNX conversion failed. Disable ONNX with: unset NLI_USE_ONNX"
                )
            
            # Get index of highest probability
            predicted_index = prob_dist.argmax()
            
            # Map to SUPPORT/REFUTE/NEUTRAL
            label = self.label_map.get(predicted_index, "NEUTRAL")
            
            result = {
                "label": label,
                "confidence": float(prob_dist[predicted_index]),
                "probabilities": {
                    "contradiction": float(prob_dist[0]), # Refute
                    "neutral": float(prob_dist[1]),       # Neutral
                    "entailment": float(prob_dist[2])     # Support
                }
            }
            results.append(result)

        return results
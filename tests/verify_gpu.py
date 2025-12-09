"""
Quick GPU verification script

Tests:
1. PyTorch CUDA availability
2. GPU memory and specifications
3. Model loading on GPU
4. Inference speed comparison (CPU vs GPU)

Usage:
    python -m tests.verify_gpu
"""

import time
import torch

def check_cuda():
    """Check CUDA availability and GPU specs"""
    print("\n" + "="*80)
    print("CUDA & GPU VERIFICATION")
    print("="*80)
    
    print("\n1. PyTorch Version:")
    print(f"   PyTorch: {torch.__version__}")
    
    print("\n2. CUDA Availability:")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA Available: True")
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
        print(f"   ✓ cuDNN Version: {torch.backends.cudnn.version()}")
        
        print("\n3. GPU Information:")
        num_gpus = torch.cuda.device_count()
        print(f"   Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"      Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"      CUDA Capability: {props.major}.{props.minor}")
            print(f"      Multi-Processor Count: {props.multi_processor_count}")
        
        print("\n4. Current GPU Memory Usage:")
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"   GPU {i}:")
            print(f"      Allocated: {allocated:.2f} GB")
            print(f"      Reserved: {reserved:.2f} GB")
        
        return True
    else:
        print(f"   ✗ CUDA Available: False")
        print("\n   Possible reasons:")
        print("   1. No NVIDIA GPU detected")
        print("   2. CUDA Toolkit not installed")
        print("   3. PyTorch installed without CUDA support")
        print("\n   To fix, install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        return False

def test_model_loading():
    """Test loading NLI model on GPU"""
    print("\n" + "="*80)
    print("MODEL LOADING TEST")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\n⚠️  GPU not available - skipping model test")
        return
    
    try:
        from src.nli.nli_model import NLIModel
        
        print("\nLoading DeBERTa-v3-base on GPU...")
        start = time.time()
        model = NLIModel(
            model_name="microsoft/deberta-v3-base",
            device="cuda"
        )
        load_time = time.time() - start
        
        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"✓ Model device: {model.device}")
        
        # Check GPU memory after loading
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✓ GPU Memory Used: {allocated:.2f} GB")
        
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

def test_inference_speed(model):
    """Compare inference speed on CPU vs GPU"""
    print("\n" + "="*80)
    print("INFERENCE SPEED TEST")
    print("="*80)
    
    if model is None:
        print("\n⚠️  No model available - skipping inference test")
        return
    
    claim = "Donald Trump is the 45th president of the U.S."
    evidence = [
        "Donald Trump served as the 45th president from 2017 to 2021.",
        "Trump was elected in 2016 and inaugurated in January 2017.",
        "He was succeeded by Joe Biden as the 46th president.",
        "Trump's presidency was marked by significant policy changes.",
        "The 45th presidency saw major tax reform legislation."
    ]
    
    print(f"\nTest Claim: {claim}")
    print(f"Evidence Sentences: {len(evidence)}")
    
    # Warm up (first inference is always slower)
    print("\n1. Warm-up inference (models need to compile kernels)...")
    _ = model.predict_batch(evidence[:1], claim)
    print("   ✓ Warm-up complete")
    
    # GPU inference
    print("\n2. GPU Inference (5 sentences)...")
    gpu_start = time.time()
    gpu_results = model.predict_batch(evidence, claim)
    gpu_time = time.time() - gpu_start
    
    print(f"   ✓ Completed in {gpu_time:.4f}s")
    print(f"   ✓ Speed: {len(evidence) / gpu_time:.1f} sentences/second")
    
    # Show results
    print("\n3. Results:")
    for i, (sent, result) in enumerate(zip(evidence, gpu_results), 1):
        print(f"   {i}. {result['label']:<8} (conf: {result['confidence']:.2f}) - {sent[:50]}...")
    
    # Memory usage
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"\n4. GPU Memory Used: {allocated:.2f} GB")
    
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"GPU Inference Time: {gpu_time:.4f}s for {len(evidence)} sentences")
    print(f"Throughput: {len(evidence) / gpu_time:.1f} sentences/second")
    print(f"Est. Time for 12 sentences: {(12 * gpu_time / len(evidence)):.4f}s")
    print(f"Memory Footprint: {allocated:.2f} GB")

def test_onnx_availability():
    """Check if ONNX Runtime GPU is available"""
    print("\n" + "="*80)
    print("ONNX RUNTIME CHECK")
    print("="*80)
    
    try:
        import onnxruntime as ort
        print(f"\n✓ ONNX Runtime installed: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print("\nAvailable Execution Providers:")
        for provider in providers:
            print(f"   - {provider}")
        
        if 'CUDAExecutionProvider' in providers:
            print("\n✓ GPU acceleration available for ONNX")
            print("  To use: $env:NLI_USE_ONNX='true'")
        else:
            print("\n⚠️  ONNX GPU support not available")
            print("  Install: pip install onnxruntime-gpu optimum[onnxruntime-gpu]")
    except ImportError:
        print("\n✗ ONNX Runtime not installed")
        print("  For CPU: pip install optimum[onnxruntime]")
        print("  For GPU: pip install onnxruntime-gpu optimum[onnxruntime-gpu]")

def main():
    print("\n" + "="*80)
    print("GPU VERIFICATION & PERFORMANCE TEST")
    print("="*80)
    
    # Step 1: Check CUDA
    has_gpu = check_cuda()
    
    # Step 2: Check ONNX
    test_onnx_availability()
    
    if has_gpu:
        # Step 3: Test model loading
        model = test_model_loading()
        
        # Step 4: Test inference speed
        if model:
            test_inference_speed(model)
        
        print("\n" + "="*80)
        print("✅ GPU SETUP COMPLETE!")
        print("="*80)
        print("\nYou can now run the fact-checking system with GPU acceleration:")
        print("  python -m src.api.api")
        print("\nFor maximum speed with ONNX:")
        print("  $env:NLI_USE_ONNX='true'")
        print("  python -m src.api.api")
    else:
        print("\n" + "="*80)
        print("❌ GPU NOT AVAILABLE")
        print("="*80)
        print("\nThe system will run on CPU (slower but functional)")
        print("\nTo enable GPU acceleration:")
        print("1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("2. Install PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Verify: python -c \"import torch; print(torch.cuda.is_available())\"")

if __name__ == "__main__":
    main()

"""
Compare different NLI models for accuracy and speed.

Usage:
    python -m tests.compare_nli_models
"""

import time
from src.nli.nli_model import NLIModel

# Test cases with expected results
test_cases = [
    {
        "claim": "Donald Trump is the 45th president of the U.S.",
        "evidence": [
            "Donald Trump served as the 45th president of the United States from 2017 to 2021.",
            "Donald Trump is the 46th president of the United States.",
            "Trump was elected in 2016 and took office in January 2017.",
        ],
        "expected": ["SUPPORT", "REFUTE", "SUPPORT"]
    },
    {
        "claim": "Python is a compiled programming language",
        "evidence": [
            "Python is an interpreted high-level programming language.",
            "Python code is compiled to bytecode before execution.",
            "The weather is sunny today.",
        ],
        "expected": ["REFUTE", "SUPPORT", "NEUTRAL"]
    },
    {
        "claim": "The Great Wall of China is visible from space",
        "evidence": [
            "The Great Wall of China is not visible from space with the naked eye.",
            "Astronauts have reported seeing the Great Wall from the ISS.",
            "The Great Wall is a famous tourist attraction in China.",
        ],
        "expected": ["REFUTE", "SUPPORT", "NEUTRAL"]
    }
]

# Models to compare
models = [
    {
        "name": "DeBERTa-v3-Large (RECOMMENDED)",
        "model_id": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        "quantization": False,
        "onnx": False
    },
    {
        "name": "DeBERTa-v3-Base (Faster)",
        "model_id": "microsoft/deberta-v3-base",
        "quantization": False,
        "onnx": False
    },
    {
        "name": "RoBERTa-Large (Original)",
        "model_id": "FacebookAI/roberta-large-mnli",
        "quantization": False,
        "onnx": False
    },
    {
        "name": "RoBERTa-Large + INT8 Quantization",
        "model_id": "FacebookAI/roberta-large-mnli",
        "quantization": True,
        "onnx": False
    },
    {
        "name": "DistilRoBERTa-Base (Fast but poor)",
        "model_id": "typeform/distilroberta-base-v2",
        "quantization": False,
        "onnx": False
    }
]

def evaluate_model(model_config):
    """Test a single model configuration"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_config['name']}")
    print(f"Model ID: {model_config['model_id']}")
    print(f"Quantization: {model_config['quantization']}, ONNX: {model_config['onnx']}")
    print('='*80)
    
    try:
        # Load model
        load_start = time.time()
        model = NLIModel(
            model_name=model_config['model_id'],
            use_quantization=model_config['quantization'],
            use_onnx=model_config['onnx']
        )
        load_time = time.time() - load_start
        print(f"⏱️  Load time: {load_time:.2f}s\n")
        
        total_correct = 0
        total_predictions = 0
        total_inference_time = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test['claim']}")
            print("-" * 80)
            
            # Run inference
            inf_start = time.time()
            results = model.predict_batch(test['evidence'], test['claim'])
            inf_time = time.time() - inf_start
            total_inference_time += inf_time
            
            print(f"⏱️  Inference time: {inf_time:.3f}s ({len(test['evidence'])} sentences)")
            print()
            
            # Check accuracy
            for j, (result, expected, evidence) in enumerate(zip(results, test['expected'], test['evidence']), 1):
                predicted = result['label']
                confidence = result['confidence']
                correct = predicted == expected
                total_correct += correct
                total_predictions += 1
                
                status = "✓" if correct else "✗"
                print(f"{status} Evidence {j}: {evidence[:60]}...")
                print(f"   Expected: {expected}, Predicted: {predicted} (conf: {confidence:.2f})")
        
        # Summary
        accuracy = (total_correct / total_predictions) * 100
        avg_time = total_inference_time / len(test_cases)
        
        print(f"\n{'='*80}")
        print(f"RESULTS for {model_config['name']}")
        print(f"{'='*80}")
        print(f"✓ Accuracy: {total_correct}/{total_predictions} ({accuracy:.1f}%)")
        print(f"⏱️  Average inference time: {avg_time:.3f}s per batch")
        print(f"⏱️  Load time: {load_time:.2f}s")
        print(f"{'='*80}\n")
        
        return {
            "name": model_config['name'],
            "accuracy": accuracy,
            "avg_inference_time": avg_time,
            "load_time": load_time,
            "correct": total_correct,
            "total": total_predictions
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    print("\n" + "="*80)
    print("NLI MODEL COMPARISON BENCHMARK")
    print("="*80)
    print(f"Testing {len(test_cases)} claim-evidence scenarios")
    print(f"Total predictions per model: {sum(len(t['evidence']) for t in test_cases)}")
    print("="*80)
    
    results = []
    for model_config in models:
        result = evaluate_model(model_config)
        if result:
            results.append(result)
        
        # Add a delay between models to let memory clear
        time.sleep(2)
    
    # Final comparison
    if results:
        print("\n" + "="*80)
        print("FINAL COMPARISON")
        print("="*80)
        print(f"{'Model':<45} {'Accuracy':<12} {'Speed':<12} {'Load Time'}")
        print("-" * 80)
        
        # Sort by accuracy descending
        results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        for r in results:
            print(f"{r['name']:<45} {r['accuracy']:>6.1f}%     {r['avg_inference_time']:>6.3f}s      {r['load_time']:>6.2f}s")
        
        print("="*80)
        print("\n🎯 RECOMMENDATION:")
        best = results[0]
        fastest = min(results, key=lambda x: x['avg_inference_time'])
        
        print(f"   Best Accuracy: {best['name']} ({best['accuracy']:.1f}%)")
        print(f"   Fastest: {fastest['name']} ({fastest['avg_inference_time']:.3f}s)")
        
        # Find best balance (accuracy > 80% and fastest)
        balanced = [r for r in results if r['accuracy'] >= 80]
        if balanced:
            best_balanced = min(balanced, key=lambda x: x['avg_inference_time'])
            print(f"   Best Balance: {best_balanced['name']} ({best_balanced['accuracy']:.1f}% @ {best_balanced['avg_inference_time']:.3f}s)")
        
        print("\n💡 To use a specific model, set environment variable:")
        print("   export NLI_MODEL_NAME=\"MoritzLaurer/deberta-v3-large-zeroshot-v2.0\"")
        print("\n💡 To enable optimizations:")
        print("   export NLI_USE_QUANTIZATION=true  # INT8 quantization (CPU only)")
        print("   export NLI_USE_ONNX=true          # ONNX Runtime (requires: pip install optimum[onnxruntime])")

if __name__ == "__main__":
    main()

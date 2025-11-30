from src.nli.batch_inference import run_nli_inference

def test_nli_module():
    # 1. Mock Claim
    claim = "Vietnam is the world's second largest coffee exporter."

    # 2. Mock Evidence (Simulating Phase 1 Output)
    mock_evidence = [
        {
            "text": "Vietnam is currently the second largest producer of coffee worldwide, behind Brazil.",
            "doc_id": 1,
            "score": 0.85
        },
        {
            "text": "Coffee production in Vietnam has declined significantly this year.",
            "doc_id": 2,
            "score": 0.70
        },
        {
            "text": "The weather in Hanoi is very humid today.",
            "doc_id": 3,
            "score": 0.20
        }
    ]

    print(f"Claim: {claim}\n")

    # 3. Run Inference
    results = run_nli_inference(claim, mock_evidence)

    # 4. Print Results
    for item in results:
        print(f"Sentence: {item['text']}")
        print(f"Label:    {item['nli_label']}")
        print(f"Conf:     {item['nli_confidence']:.4f}")
        print(f"Probs:    {item['nli_probs']}")
        print("-" * 50)

if __name__ == "__main__":
    test_nli_module()
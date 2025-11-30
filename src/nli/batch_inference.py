import logging
from typing import List, Dict, Any
from .nli_model import NLIModel

logger = logging.getLogger(__name__)

# Singleton instance to avoid reloading model across calls
_NLI_MODEL_INSTANCE = None

def get_model_instance():
    """Singleton accessor for the NLI model."""
    global _NLI_MODEL_INSTANCE
    if _NLI_MODEL_INSTANCE is None:
        # UPDATED: Use the correct official ID
        _NLI_MODEL_INSTANCE = NLIModel(model_name="FacebookAI/roberta-large-mnli")
    return _NLI_MODEL_INSTANCE

def run_nli_inference(claim: str, ranked_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Main entry point for Phase 2.
    Takes the claim and the top ranked sentences from Phase 1, 
    runs NLI, and appends the results to the evidence objects.

    Args:
        claim (str): The user's claim to verify.
        ranked_evidence (list): List of dicts (output from Phase 1 retrieval).
                                Expected format: {'text': '...', 'doc_id': ...}

    Returns:
        list: The same evidence list, enriched with 'nli_label' and 'nli_score'.
    """
    if not ranked_evidence:
        logger.warning("No evidence provided for NLI inference.")
        return []

    logger.info(f"Running NLI on {len(ranked_evidence)} sentences for claim: '{claim}'")
    
    model = get_model_instance()

    # Extract just the text for the model
    premises = [item.get('text', '') for item in ranked_evidence]
    
    # Run batch prediction
    try:
        predictions = model.predict_batch(premises, claim)
    except Exception as e:
        logger.error(f"NLI Inference failed: {e}")
        return ranked_evidence # Return original list if failure occurs

    # Merge predictions back into the evidence objects
    enriched_evidence = []
    for original_item, pred in zip(ranked_evidence, predictions):
        new_item = original_item.copy()
        
        new_item['nli_label'] = pred['label']          
        new_item['nli_confidence'] = pred['confidence'] 
        new_item['nli_probs'] = pred['probabilities']   
        
        enriched_evidence.append(new_item)
        
        logger.debug(f"Sentence: {new_item['text'][:30]}... -> {pred['label']} ({pred['confidence']:.2f})")

    return enriched_evidence
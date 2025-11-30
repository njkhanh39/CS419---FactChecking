import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLIModel:
    """
    Wrapper for RoBERTa-MNLI model to perform Natural Language Inference.
    Determines if a premise (evidence) supports, refutes, or is neutral to a hypothesis (claim).
    """

    # UPDATED: Use the official repository ID. 
    # If this is too slow on your machine, try "cross-encoder/nli-distilroberta-base"
    # but note that label mappings might differ for other models.
    def __init__(self, model_name="FacebookAI/roberta-large-mnli", device=None):
        """
        Initialize the NLI model.
        
        Args:
            model_name (str): HuggingFace model name (default: FacebookAI/roberta-large-mnli)
            device (str): 'cuda', 'mps', or 'cpu'. If None, auto-detects.
        """
        self.model_name = model_name
        
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
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {e}")
            raise e

        # Standard RoBERTa-MNLI mapping for FacebookAI models:
        # 0 = Contradiction (REFUTE)
        # 1 = Neutral (NEUTRAL)
        # 2 = Entailment (SUPPORT)
        self.label_map = {0: "REFUTE", 1: "NEUTRAL", 2: "SUPPORT"}

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
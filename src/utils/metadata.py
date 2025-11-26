"""
Metadata Handler: Extract, score, and manage document/sentence metadata
Supports metadata-based ranking and filtering
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dateutil import parser as date_parser
import re
from urllib.parse import urlparse


class MetadataHandler:
    """
    Handles metadata extraction, scoring, and relevance assessment
    """
    
    def __init__(self, trusted_domains: Optional[List[str]] = None):
        """
        Initialize metadata handler
        
        Args:
            trusted_domains: List of trusted news sources for authority scoring
        """
        self.trusted_domains = trusted_domains or [
            # International
            "reuters.com", "apnews.com", "bbc.com", "cnn.com", 
            "nytimes.com", "washingtonpost.com", "theguardian.com",
            # Vietnamese
            "vtv.vn", "vnexpress.net", "tuoitre.vn", "thanhnien.vn",
            # Scientific
            "nature.com", "science.org", "nih.gov", "who.int",
            # Other
            "wikipedia.org", "gov", "edu"
        ]
    
    def parse_date(self, date_string: str) -> Optional[datetime]:
        """
        Parse various date formats into datetime object
        
        Args:
            date_string: Date string in various formats
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_string:
            return None
        
        try:
            # Handle ISO format and common formats
            return date_parser.parse(date_string, fuzzy=True)
        except:
            return None
    
    def calculate_recency_score(
        self, 
        article_date: str,
        claim_date: Optional[str] = None,
        max_days: int = 365
    ) -> float:
        """
        Calculate recency score based on article publication date
        
        Args:
            article_date: Article publication date
            claim_date: Reference date (defaults to current date)
            max_days: Maximum days to consider relevant
            
        Returns:
            Score between 0.0 and 1.0 (1.0 = most recent)
        """
        article_dt = self.parse_date(article_date)
        if not article_dt:
            return 0.5  # Neutral score if date unknown
        
        if claim_date:
            reference_dt = self.parse_date(claim_date)
        else:
            reference_dt = datetime.now()
        
        if not reference_dt:
            reference_dt = datetime.now()
        
        # Calculate days difference
        days_diff = abs((reference_dt - article_dt).days)
        
        # Score decreases with age (exponential decay)
        # Articles within 1 year are most relevant
        if days_diff <= max_days:
            score = 1.0 - (days_diff / max_days) ** 0.5
        else:
            score = 0.2  # Old articles get low but non-zero score
        
        return max(0.0, min(1.0, score))
    
    def calculate_authority_score(self, url: str, domain: Optional[str] = None) -> float:
        """
        Calculate source authority score based on domain
        
        Args:
            url: Article URL
            domain: Domain name (extracted if not provided)
            
        Returns:
            Score between 0.0 and 1.0 (1.0 = most trusted)
        """
        if not domain:
            domain = urlparse(url).netloc.lower()
        
        # Check if domain is in trusted list
        for trusted in self.trusted_domains:
            if trusted in domain:
                return 1.0
        
        # Check for common quality indicators
        if any(tld in domain for tld in [".gov", ".edu", ".org"]):
            return 0.8
        
        # Default score for unknown sources
        return 0.5
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Simple entity extraction (dates, numbers, capitalized phrases)
        For production, use spaCy or similar NER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "dates": [],
            "numbers": [],
            "capitalized_phrases": []
        }
        
        # Extract years (4-digit numbers)
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        entities["dates"].extend(years)
        
        # Extract numbers with units
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|%|percent)?\b', text)
        entities["numbers"].extend(numbers)
        
        # Extract capitalized phrases (potential named entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities["capitalized_phrases"].extend(capitalized[:10])  # Limit to avoid noise
        
        return entities
    
    def calculate_entity_overlap_score(
        self, 
        claim_entities: Dict[str, List[str]], 
        sentence_entities: Dict[str, List[str]]
    ) -> float:
        """
        Calculate entity overlap between claim and sentence
        
        Args:
            claim_entities: Entities from claim
            sentence_entities: Entities from sentence
            
        Returns:
            Score between 0.0 and 1.0
        """
        total_score = 0.0
        weight_sum = 0.0
        
        # Weight different entity types
        weights = {
            "dates": 0.4,
            "numbers": 0.3,
            "capitalized_phrases": 0.3
        }
        
        for entity_type, weight in weights.items():
            claim_ents = set(claim_entities.get(entity_type, []))
            sent_ents = set(sentence_entities.get(entity_type, []))
            
            if claim_ents:
                overlap = len(claim_ents & sent_ents) / len(claim_ents)
                total_score += overlap * weight
                weight_sum += weight
        
        return total_score / weight_sum if weight_sum > 0 else 0.5
    
    def calculate_metadata_score(
        self,
        document: Dict,
        claim: str,
        claim_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metadata score for a document
        
        Args:
            document: Document with metadata
            claim: User's claim
            claim_date: Date associated with claim
            
        Returns:
            Dictionary of metadata scores
        """
        scores = {
            "recency": 0.5,
            "authority": 0.5,
            "entity_overlap": 0.5,
            "combined": 0.5
        }
        
        # Recency score
        if document.get("date"):
            scores["recency"] = self.calculate_recency_score(
                document["date"], 
                claim_date
            )
        
        # Authority score
        if document.get("url"):
            scores["authority"] = self.calculate_authority_score(
                document["url"],
                document.get("domain")
            )
        
        # Entity overlap score
        claim_entities = self.extract_entities(claim)
        doc_entities = self.extract_entities(document.get("text", "")[:1000])  # Sample
        scores["entity_overlap"] = self.calculate_entity_overlap_score(
            claim_entities,
            doc_entities
        )
        
        # Combined metadata score (weighted average)
        scores["combined"] = (
            scores["recency"] * 0.3 +
            scores["authority"] * 0.4 +
            scores["entity_overlap"] * 0.3
        )
        
        return scores
    
    def rank_documents_by_metadata(
        self,
        documents: List[Dict],
        claim: str,
        claim_date: Optional[str] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Rank documents by metadata scores
        
        Args:
            documents: List of documents
            claim: User's claim
            claim_date: Date associated with claim
            
        Returns:
            List of (document, score) tuples sorted by score
        """
        scored_docs = []
        
        for doc in documents:
            metadata_scores = self.calculate_metadata_score(doc, claim, claim_date)
            doc_with_scores = doc.copy()
            doc_with_scores["metadata_scores"] = metadata_scores
            scored_docs.append((doc_with_scores, metadata_scores["combined"]))
        
        # Sort by combined score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs


# Example usage
if __name__ == "__main__":
    handler = MetadataHandler()
    
    # Example document
    doc = {
        "url": "https://www.reuters.com/article/coffee-exports",
        "text": "Vietnam exported 1.7 million tons of coffee in 2023...",
        "date": "2023-12-15",
        "domain": "reuters.com"
    }
    
    claim = "Vietnam is the world's second largest coffee exporter in 2023"
    
    scores = handler.calculate_metadata_score(doc, claim)
    print("Metadata Scores:")
    print(f"  Recency: {scores['recency']:.2f}")
    print(f"  Authority: {scores['authority']:.2f}")
    print(f"  Entity Overlap: {scores['entity_overlap']:.2f}")
    print(f"  Combined: {scores['combined']:.2f}")

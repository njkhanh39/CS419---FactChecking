"""
Data Collection Orchestrator: Complete pipeline from claim to corpus
Combines web search and scraping into a single workflow
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from .web_search import WebSearcher
from .web_scraper import WebScraper


class DataCollector:
    """
    Orchestrates the complete data collection pipeline:
    Claim → Search → Scrape → Save Corpus
    """
    
    def __init__(
        self, 
        search_api: str = "serpapi",
        api_key: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the data collector
        
        Args:
            search_api: Search API to use ("serpapi" or "bing")
            api_key: API key for the search service
            output_dir: Directory to save collected data
        """
        self.searcher = WebSearcher(api_type=search_api, api_key=api_key)
        self.scraper = WebScraper()
        
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to data/raw/ in project root
            from ..config.paths import DATA_RAW_DIR
            self.output_dir = Path(DATA_RAW_DIR)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_corpus(
        self, 
        claim: str,
        num_urls: int = 20,
        save: bool = True,
        filename: Optional[str] = None
    ) -> Dict:
        """
        Complete data collection pipeline for a claim
        
        Args:
            claim: The claim to fact-check
            num_urls: Number of URLs to retrieve and scrape
            save: Whether to save the corpus to disk
            filename: Custom filename for saved corpus
            
        Returns:
            Dictionary containing claim, search results, and corpus
        """
        print(f"\n{'='*60}")
        print(f"DATA COLLECTION PIPELINE")
        print(f"{'='*60}")
        print(f"Claim: {claim}")
        print(f"Target URLs: {num_urls}\n")
        
        # Step 1: Web Search
        print("[1/3] Searching the web...")
        search_results = self.searcher.search(claim, num_results=num_urls)
        print(f"      Found {len(search_results)} URLs")
        
        if not search_results:
            print("      ✗ No search results found")
            return {"claim": claim, "search_results": [], "corpus": [], "metadata": {}}
        
        # Step 2: Web Scraping
        print(f"\n[2/3] Scraping {len(search_results)} URLs...")
        documents = self.scraper.scrape_from_search_results(search_results, max_documents=num_urls)
        print(f"      Successfully scraped {len(documents)} documents")
        
        # Step 3: Prepare corpus with metadata
        print(f"\n[3/3] Preparing corpus...")
        corpus = {
            "claim": claim,
            "search_results": search_results,
            "corpus": documents,
            "metadata": {
                "collection_date": datetime.now().isoformat(),
                "num_search_results": len(search_results),
                "num_documents_scraped": len(documents),
                "search_engine": search_results[0].get("search_engine", "unknown") if search_results else "unknown"
            }
        }
        
        # Step 4: Save to disk
        if save:
            if not filename:
                # Generate filename from claim and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_claim = "".join(c if c.isalnum() else "_" for c in claim[:50])
                filename = f"corpus_{safe_claim}_{timestamp}.json"
            
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False)
            
            print(f"      ✓ Saved to: {filepath}")
        
        print(f"\n{'='*60}")
        print(f"COLLECTION COMPLETE")
        print(f"{'='*60}\n")
        
        return corpus
    
    def load_corpus(self, filename: str) -> Dict:
        """
        Load a previously saved corpus
        
        Args:
            filename: Name of the corpus file
            
        Returns:
            Loaded corpus dictionary
        """
        filepath = self.output_dir / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_saved_corpora(self) -> List[str]:
        """
        List all saved corpus files
        
        Returns:
            List of corpus filenames
        """
        return [f.name for f in self.output_dir.glob("corpus_*.json")]
    
    def get_corpus_summary(self, corpus: Dict) -> Dict:
        """
        Get summary statistics for a corpus
        
        Args:
            corpus: Corpus dictionary
            
        Returns:
            Summary statistics
        """
        documents = corpus.get("corpus", [])
        
        total_chars = sum(len(doc.get("text", "")) for doc in documents)
        total_words = sum(len(doc.get("text", "").split()) for doc in documents)
        
        return {
            "claim": corpus.get("claim", ""),
            "num_documents": len(documents),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chars_per_doc": total_chars // len(documents) if documents else 0,
            "avg_words_per_doc": total_words // len(documents) if documents else 0,
            "collection_date": corpus.get("metadata", {}).get("collection_date", "unknown"),
            "sources": list(set(doc.get("domain", "") for doc in documents))
        }


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = DataCollector(search_api="serpapi")
    
    # Collect data for a claim
    claim = "Vietnam is the world's second largest coffee producer"
    corpus = collector.collect_corpus(claim, num_urls=5, save=True)
    
    # Print summary
    summary = collector.get_corpus_summary(corpus)
    print("\nCorpus Summary:")
    print(f"  Claim: {summary['claim']}")
    print(f"  Documents: {summary['num_documents']}")
    print(f"  Total words: {summary['total_words']:,}")
    print(f"  Avg words/doc: {summary['avg_words_per_doc']:,}")
    print(f"  Sources: {', '.join(summary['sources'][:5])}")

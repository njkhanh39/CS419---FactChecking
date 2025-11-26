"""
Web Search Module: Query Generation and Web Search API Integration
Supports Google Search via SerpApi and Bing Search API
"""

import os
from typing import List, Dict, Optional
import requests
from datetime import datetime


class WebSearcher:
    """
    Handles web search operations using various search APIs
    """
    
    def __init__(self, api_type: str = "serpapi", api_key: Optional[str] = None):
        """
        Initialize the web searcher
        
        Args:
            api_type: Type of search API ("serpapi" or "bing")
            api_key: API key for the search service
        """
        self.api_type = api_type.lower()
        self.api_key = api_key or os.getenv(f"{api_type.upper()}_API_KEY")
        
        if not self.api_key:
            print(f"Warning: No API key found for {api_type}. Set {api_type.upper()}_API_KEY environment variable.")
    
    def generate_search_queries(self, claim: str, num_variations: int = 1) -> List[str]:
        """
        Generate search queries from a claim
        For now, returns the claim as-is, but can be extended with:
        - Query expansion
        - Paraphrasing
        - Adding context keywords
        
        Args:
            claim: The user's claim to fact-check
            num_variations: Number of query variations to generate
            
        Returns:
            List of search queries
        """
        queries = [claim]
        
        # Optional: Add variations
        # For example: add "fact check" or "news" keywords
        if num_variations > 1:
            queries.append(f"{claim} fact check")
        if num_variations > 2:
            queries.append(f"{claim} news")
            
        return queries[:num_variations]
    
    def search_serpapi(self, query: str, num_results: int = 20) -> List[Dict]:
        """
        Search using SerpApi (Google Search)
        
        Args:
            query: Search query string
            num_results: Number of results to retrieve
            
        Returns:
            List of search results with URLs and metadata
        """
        if not self.api_key:
            print("Error: SerpApi key not configured")
            return []
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "engine": "google"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "url": item.get("link"),
                    "title": item.get("title"),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                    "search_engine": "google"
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching with SerpApi: {e}")
            return []
    
    def search_bing(self, query: str, num_results: int = 20) -> List[Dict]:
        """
        Search using Bing Search API
        
        Args:
            query: Search query string
            num_results: Number of results to retrieve
            
        Returns:
            List of search results with URLs and metadata
        """
        if not self.api_key:
            print("Error: Bing API key not configured")
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results,
            "responseFilter": "Webpages"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("webPages", {}).get("value", [])[:num_results]:
                results.append({
                    "url": item.get("url"),
                    "title": item.get("name"),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("displayUrl", ""),
                    "date": item.get("dateLastCrawled", ""),
                    "search_engine": "bing"
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching with Bing: {e}")
            return []
    
    def search(self, claim: str, num_results: int = 20) -> List[Dict]:
        """
        Main search interface - automatically uses configured API
        
        Args:
            claim: The claim to search for
            num_results: Number of results to retrieve
            
        Returns:
            List of search results
        """
        queries = self.generate_search_queries(claim)
        query = queries[0]  # Use the first (main) query
        
        if self.api_type == "serpapi":
            return self.search_serpapi(query, num_results)
        elif self.api_type == "bing":
            return self.search_bing(query, num_results)
        else:
            print(f"Error: Unknown API type '{self.api_type}'")
            return []
    
    def search_multiple_queries(self, claim: str, num_queries: int = 2, results_per_query: int = 10) -> List[Dict]:
        """
        Search using multiple query variations and combine results
        
        Args:
            claim: The claim to search for
            num_queries: Number of query variations to use
            results_per_query: Results per query
            
        Returns:
            Combined list of unique search results
        """
        queries = self.generate_search_queries(claim, num_queries)
        all_results = []
        seen_urls = set()
        
        for query in queries:
            if self.api_type == "serpapi":
                results = self.search_serpapi(query, results_per_query)
            elif self.api_type == "bing":
                results = self.search_bing(query, results_per_query)
            else:
                continue
            
            # Deduplicate by URL
            for result in results:
                url = result.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
        
        return all_results


# Example usage
if __name__ == "__main__":
    # Example with SerpApi
    searcher = WebSearcher(api_type="serpapi")
    claim = "Vietnam is the world's largest coffee exporter"
    
    print(f"Searching for: {claim}")
    results = searcher.search(claim, num_results=5)
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")

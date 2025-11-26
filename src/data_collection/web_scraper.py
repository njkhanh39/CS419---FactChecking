"""
Web Scraper Module: Downloads and extracts text from web pages
Uses trafilatura (primary) and newspaper3k (fallback) for robust extraction
"""

import trafilatura
from newspaper import Article
import requests
from typing import Dict, List, Optional
from datetime import datetime
import time
from urllib.parse import urlparse


class WebScraper:
    """
    Handles web scraping and text extraction from URLs
    """
    
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize the web scraper
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
    
    def scrape_with_trafilatura(self, url: str) -> Optional[Dict]:
        """
        Scrape webpage using trafilatura (primary method)
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with extracted content and metadata, or None if failed
        """
        try:
            # Download HTML
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            
            # Extract text content
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False
            )
            
            if not text:
                return None
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            return {
                "url": url,
                "text": text,
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "description": metadata.description if metadata else "",
                "sitename": metadata.sitename if metadata else "",
                "extraction_method": "trafilatura",
                "scraped_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Trafilatura failed for {url}: {e}")
            return None
    
    def scrape_with_newspaper(self, url: str) -> Optional[Dict]:
        """
        Scrape webpage using newspaper3k (fallback method)
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with extracted content and metadata, or None if failed
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text:
                return None
            
            # Handle publish_date - it can be datetime or string
            date_str = ""
            if article.publish_date:
                if isinstance(article.publish_date, datetime):
                    date_str = article.publish_date.isoformat()
                else:
                    date_str = str(article.publish_date)
            
            return {
                "url": url,
                "text": article.text,
                "title": article.title or "",
                "author": ", ".join(article.authors) if article.authors else "",
                "date": date_str,
                "description": article.meta_description or "",
                "sitename": urlparse(url).netloc,
                "extraction_method": "newspaper3k",
                "scraped_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Newspaper3k failed for {url}: {e}")
            return None
    
    def scrape_url(self, url: str, use_fallback: bool = True) -> Optional[Dict]:
        """
        Scrape a single URL with automatic fallback
        
        Args:
            url: URL to scrape
            use_fallback: Whether to try fallback method if primary fails
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Try trafilatura first (faster and more reliable)
        result = self.scrape_with_trafilatura(url)
        
        # Fallback to newspaper3k if trafilatura fails
        if not result and use_fallback:
            result = self.scrape_with_newspaper(url)
        
        if result:
            # Add domain information
            result["domain"] = urlparse(url).netloc
            
        return result
    
    def scrape_multiple_urls(
        self, 
        urls: List[str], 
        delay: float = 1.0,
        max_retries: int = 2
    ) -> List[Dict]:
        """
        Scrape multiple URLs with rate limiting and retry logic
        
        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds
            max_retries: Maximum number of retry attempts per URL
            
        Returns:
            List of successfully scraped documents
        """
        documents = []
        
        for i, url in enumerate(urls):
            print(f"Scraping {i+1}/{len(urls)}: {url}")
            
            # Try scraping with retries
            result = None
            for attempt in range(max_retries + 1):
                result = self.scrape_url(url)
                if result:
                    break
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
            
            if result:
                documents.append(result)
                print(f"  ✓ Success: {len(result['text'])} characters")
            else:
                print(f"  ✗ Failed to scrape")
            
            # Rate limiting (except for last URL)
            if i < len(urls) - 1:
                time.sleep(delay)
        
        print(f"\nSuccessfully scraped {len(documents)}/{len(urls)} URLs")
        return documents
    
    def scrape_from_search_results(
        self, 
        search_results: List[Dict],
        max_documents: int = 20
    ) -> List[Dict]:
        """
        Scrape documents from search results
        
        Args:
            search_results: List of search results from web_search.py
            max_documents: Maximum number of documents to scrape
            
        Returns:
            List of scraped documents with content and metadata
        """
        urls = [result["url"] for result in search_results[:max_documents] if result.get("url")]
        documents = self.scrape_multiple_urls(urls)
        
        # Merge search metadata with scraped content
        for doc in documents:
            # Find matching search result
            matching_result = next(
                (r for r in search_results if r["url"] == doc["url"]), 
                None
            )
            if matching_result:
                doc["search_title"] = matching_result.get("title", "")
                doc["search_snippet"] = matching_result.get("snippet", "")
                doc["search_engine"] = matching_result.get("search_engine", "")
        
        return documents


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    
    # Test single URL
    test_url = "https://en.wikipedia.org/wiki/Coffee_production_in_Vietnam"
    print(f"Testing URL: {test_url}\n")
    
    result = scraper.scrape_url(test_url)
    if result:
        print(f"Title: {result['title']}")
        print(f"Date: {result['date']}")
        print(f"Method: {result['extraction_method']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"\nFirst 200 chars:\n{result['text'][:200]}...")
    else:
        print("Failed to scrape")

"""
Data Collection Module

This module handles the complete data collection pipeline for fact-checking:
1. Query Generation: Generate search queries from user claims
2. Web Search: Search Google/Bing for relevant URLs
3. Web Scraping: Extract content from web pages
4. Corpus Creation: Build a collection of documents with metadata

Components:
-----------
- web_search.py: Search API integration (SerpApi, Bing)
- web_scraper.py: Web scraping (trafilatura, newspaper3k)
- collector.py: Complete pipeline orchestrator

Usage:
------
from src.data_collection.collector import DataCollector

collector = DataCollector(search_api="serpapi")
corpus = collector.collect_corpus(claim="Your claim here", num_urls=20)
"""

from .collector import DataCollector
from .web_search import WebSearcher
from .web_scraper import WebScraper

__all__ = ['DataCollector', 'WebSearcher', 'WebScraper']

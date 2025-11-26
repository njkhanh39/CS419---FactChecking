"""
Configuration file for API keys and settings
Copy this file to api_keys.py and add your actual API keys
(api_keys.py is in .gitignore for security)
"""

# Search API Configuration
SERPAPI_KEY = "your_serpapi_key_here"  # Get from https://serpapi.com/
BING_API_KEY = "your_bing_api_key_here"  # Get from Azure Portal

# Default search settings
DEFAULT_SEARCH_API = "serpapi"  # Options: "serpapi" or "bing"
DEFAULT_NUM_URLS = 20

# Scraping settings
SCRAPING_TIMEOUT = 30  # seconds
SCRAPING_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 2

# Trusted domains for authority scoring
TRUSTED_DOMAINS = [
    # International news
    "reuters.com", "apnews.com", "bbc.com", "cnn.com",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    # Vietnamese news
    "vtv.vn", "vnexpress.net", "tuoitre.vn", "thanhnien.vn",
    # Scientific sources
    "nature.com", "science.org", "nih.gov", "who.int",
    # Government and education
    "gov", "edu", "wikipedia.org"
]

# Retrieval settings
BM25_K1 = 1.5
BM25_B = 0.75
NUM_CANDIDATE_SENTENCES = 50  # Stage 1: BM25 output
NUM_FINAL_SENTENCES = 10      # Stage 2: Final ranked sentences

# Scoring weights for sentence ranking
SEMANTIC_WEIGHT = 0.5
LEXICAL_WEIGHT = 0.3
METADATA_WEIGHT = 0.2

# NLI settings
NLI_MODEL_NAME = "roberta-large-mnli"  # or "roberta-base-mnli"
NLI_BATCH_SIZE = 8

# Aggregation thresholds
SUPPORT_THRESHOLD = 0.5    # S_final > 0.5 → SUPPORTED
REFUTE_THRESHOLD = -0.5    # S_final < -0.5 → REFUTED
# Otherwise → INSUFFICIENT EVIDENCE

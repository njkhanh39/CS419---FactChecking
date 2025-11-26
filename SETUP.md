# Setup Guide

## Prerequisites
- Python 3.8+
- pip package manager
- API keys for web search (SerpApi or Bing)

## Installation Steps

### 1. Clone/Navigate to Project
```bash
cd "E:\File\Code\Stuff Files\CS419 - IR\CS419---FactChecking"
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
# SERPAPI_KEY=your_key_here
# BING_SEARCH_KEY=your_key_here
```

### 6. (Optional) Download NLI Model for ONNX
If you want to use ONNX for faster inference:
1. Download a pre-converted ONNX model (e.g., roberta-base-mnli)
2. Place it in `models/roberta-base-mnli-onnx/`
3. Update `NLI_MODEL_PATH` in `.env`

Otherwise, the system will fallback to using transformers directly.

## Quick Start

### Run Fact-Checking
```bash
python main.py --claim "Your claim here"
```

### Examples
```bash
# Using SerpApi (default)
python main.py --claim "The Earth is flat"

# Using Bing Search
python main.py --claim "COVID-19 vaccines are effective" --engine bing

# Save results to JSON
python main.py --claim "Python is faster than Java" --output result.json

# Verbose mode
python main.py --claim "Climate change is real" --verbose
```

## Troubleshooting

### API Key Errors
- Make sure you've set `SERPAPI_KEY` or `BING_SEARCH_KEY` in `.env`
- Verify your API key is valid and has remaining credits

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Not Found
The system will auto-download required NLTK data on first run. If issues persist:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Memory Issues
- Reduce `MAX_URLS` in `.env` (default: 20)
- Reduce `BM25_TOP_K` (default: 50)
- Use a smaller embedding model

## Next Steps

After setup, you can:
1. Test the system with sample claims
2. Customize configuration in `.env`
3. Implement custom query expansion strategies
4. Add more domain authority sources
5. Fine-tune scoring weights
6. Integrate with a web interface

## Development

### Project Structure
See `STRUCTURE.md` for detailed module descriptions.

### Testing
```bash
# Test web search
python -c "from src.crawling.search import WebSearcher; ws = WebSearcher(); print(ws.search('test', 5))"

# Test scraping
python -c "from src.crawling.scraper import ContentScraper; cs = ContentScraper(); print(cs.scrape_url('https://example.com'))"
```

### Adding Features
- Custom query expansion: Edit `src/crawling/search.py`
- Custom scoring: Edit `src/retrieval/reranker.py`
- Custom NLI models: Edit `src/inference/nli_model.py`

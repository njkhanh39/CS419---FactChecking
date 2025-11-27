# Installing Packages in Virtual Environment (ltdsword)

## Step 1: Activate your virtual environment

```powershell
# Navigate to your project directory
cd "e:\File\Code\Stuff Files\CS419 - IR\CS419---FactChecking"

# Activate the virtual environment (PowerShell)
.\ltdsword\Scripts\Activate.ps1

# If you get execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Step 2: Upgrade pip (recommended)

```powershell
python -m pip install --upgrade pip
```

## Step 3: Install all required packages

```powershell
pip install -r requirements.txt
```

This will install all the packages listed in `requirements.txt`:
- numpy, pandas, python-dateutil
- rank-bm25, faiss-cpu
- sentence-transformers, transformers, torch, tf-keras
- onnxruntime
- requests, beautifulsoup4, trafilatura, newspaper3k, lxml
- google-search-results
- pathlib, tqdm, regex

## Step 4: Verify installation

```powershell
# Check installed packages
pip list

# Test imports
python -c "import numpy, pandas, faiss, torch, transformers; print('✓ All core packages loaded successfully')"
```

## Troubleshooting

### If faiss-cpu fails to install:
```powershell
# Try installing faiss separately first
pip install faiss-cpu --no-cache-dir
```

### If torch installation is slow:
```powershell
# Install from PyTorch official repository
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### If tf-keras causes issues:
```powershell
# Install TensorFlow first
pip install tensorflow
pip install tf-keras
```

### If any package fails:
```powershell
# Install packages one by one to identify the problem
pip install numpy pandas python-dateutil
pip install rank-bm25
pip install faiss-cpu
pip install sentence-transformers
pip install transformers
pip install torch
pip install tf-keras
pip install onnxruntime
pip install requests beautifulsoup4 trafilatura newspaper3k lxml
pip install google-search-results
pip install tqdm regex
```

## Deactivate virtual environment when done

```powershell
deactivate
```

## Quick Reference

**Activate:**
```powershell
.\ltdsword\Scripts\Activate.ps1
```

**Install packages:**
```powershell
pip install -r requirements.txt
```

**Deactivate:**
```powershell
deactivate
```

# LLM Upgrade Guide

## Overview

This document describes the upgrade from LLaMA-2 to Google's Gemma model in the Medical ChatAI system.

## Changes Made

### 1. Model Architecture
- **Before**: LLaMA-2 7B using CTransformers with quantized `.bin` files
- **After**: Google Gemma 2B instruction-tuned using HuggingFace Transformers

### 2. Dependencies Updated
- Added `langchain-community` for updated LangChain components
- Updated imports to use non-deprecated LangChain modules
- Added `chainlit` to requirements.txt

### 3. Enhanced Features
- **Fallback Models**: Automatic fallback to smaller models if Gemma is unavailable
- **GPU/CPU Support**: Dynamic device detection and optimization
- **Better Integration**: Native HuggingFace ecosystem integration

## Model Selection Rationale

### Why Gemma over LLaMA-2?
1. **More Recent**: Gemma is a newer model with improved architecture
2. **Instruction-Tuned**: Better for conversational AI applications
3. **Efficient**: Gemma 2B provides good performance with lower resource requirements
4. **Open Source**: Maintained by Google with ongoing support

### Model Hierarchy
1. **Primary**: `google/gemma-2b-it` - Instruction-tuned Gemma 2B
2. **Fallback 1**: `microsoft/DialoGPT-medium` - Conversational model
3. **Fallback 2**: `gpt2` - Basic text generation

## Usage

### First-Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create vector database (requires internet for embeddings)
python ingest.py

# Run the application
chainlit run model.py -w
```

### Configuration

The model selection is automatic, but you can modify the model choice in `model.py`:
```python
# In load_llm() function
model_name = "google/gemma-2b-it"  # Change this for different models
```

## Compatibility

The upgrade maintains 100% backward compatibility:
- Same function signatures
- Same prompt template format
- Same RAG pipeline interface
- Same Chainlit web interface

## Performance Considerations

### Memory Requirements
- **Gemma 2B**: ~4-6GB RAM (recommended)
- **GPU**: Optional but recommended for faster inference
- **CPU**: Fallback supported with slower performance

### First Run
- Models download automatically from HuggingFace Hub
- Subsequent runs use cached models
- Internet required only for initial setup

## Troubleshooting

### Model Loading Issues
1. **Network Error**: Models download on first use - ensure internet connection
2. **Memory Error**: Try GPU or reduce batch size
3. **Import Error**: Run `pip install -U langchain-community`

### Performance Issues
1. Use GPU if available: `torch.cuda.is_available()`
2. Adjust `max_new_tokens` in `load_llm()` function
3. Consider model quantization for lower-spec hardware

## Testing

Run the integration tests to verify the upgrade:
```bash
PYTHONPATH=. python tests/test_integration.py
```

## Migration Notes

### For Existing Users
- Remove old LLaMA-2 `.bin` files (no longer needed)
- Update requirements: `pip install -r requirements.txt`
- No code changes needed in existing usage

### For Developers
- Import paths updated to use `langchain-community`
- Model loading now uses HuggingFace Transformers directly
- Pipeline creation includes better error handling
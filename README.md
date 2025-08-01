# AI Medical Consultant

## Overview

The AI Medical Consultant is an advanced chatbot designed to provide accurate medical information by querying a robust knowledge base derived from "The GALE Encyclopedia of Medicine." This project utilizes cutting-edge NLP techniques and frameworks to deliver reliable and instant medical guidance.

## Technical Architecture

### Data Ingestion

- **Vector Database Creation:** The textual data from "The GALE Encyclopedia of Medicine" PDF is transformed into a searchable vector database using FAISS. This process is facilitated by the Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` model, optimized for running on CPU environments.
- **Storage:** The generated embeddings are stored in a structured directory for efficient retrieval.

### Model Configuration

- **Language Model:** The project employs Google's Gemma model, specifically `google/gemma-2b-it` (instruction-tuned version), using HuggingFace Transformers. This provides more powerful and up-to-date natural language processing capabilities compared to the previous LLaMA-2 implementation.
- **Fallback Models:** The system includes fallback models for scenarios where the primary Gemma model is unavailable, ensuring robust operation.
- **Custom Prompting:** A tailored prompt template enhances the query handling by structuring user inputs and model responses effectively.

### Interface and Interaction

- **Data Chunking:** Utilizing `RecursiveCharacterTextSplitter` from the Langchain framework, the data is segmented into manageable chunks that ensure comprehensive coverage of the medical encyclopedia content.
- **Retrieval-Augmented Generation:** A `RetrievalQA` chain enhances the chatbot's response accuracy by dynamically sourcing relevant information from the vector database.

### Application

- **Web Interface:** The chatbot is accessible via a web-based application built using Chainlit, offering users a seamless and interactive platform to inquire about medical knowledge.

## Setup and Deployment

### Requirements

- Python 3.8+
- Libraries: Langchain, HuggingFace Transformers, FAISS, Chainlit, sentence_transformers
- GPU recommended for optimal performance with Gemma model (CPU fallback available)
- Internet connection required for initial model download from HuggingFace Hub

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-link]

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the ingest.py file first: to create vector database
   ```bash
   python ingest.py

4. Launch the application on localhost:
   ```bash
   chainlit run model.py -w


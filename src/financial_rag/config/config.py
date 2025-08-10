# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configuration
VLM_MODEL_ID = "Salesforce/blip-image-captioning-base"
EMBEDDING_MODEL_ID = "models/embedding-001" 
GENERATOR_MODEL_ID = "models/gemini-2.5-pro"

# Retrieval Configuration
RETRIEVER_K = 10
MAX_NEW_TOKENS = 2048

# File paths
INPUT_FILE_PATH = "data/documents/2022 Q3 AAPL.pdf"
IMAGE_OUTPUT_DIR = "data/extracted_images"
VECTORSTORE_PATH = "data/vectorstore_index"

# Table processing configuration
TABLE_EXTRACTION_STRATEGY = "hi_res"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
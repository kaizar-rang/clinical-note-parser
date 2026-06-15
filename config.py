"""
Global config. Copy .env.example to .env and fill in values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "anthropic" | "local"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "")

# Data paths
MRI_CSV = os.getenv("MRI_CSV", "data/mri_reports.csv")
EEG_CSV = os.getenv("EEG_CSV", "data/eeg_reports.csv")
CLINICAL_CSV = os.getenv("CLINICAL_CSV", "data/clinical_notes.csv")

# Splitter
MIN_PARAGRAPH_LENGTH = 30

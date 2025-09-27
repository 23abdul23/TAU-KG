"""
Utility functions for the biomedical QA system
"""
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file with fallback defaults
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        "pubmed": {
            "max_retries": 3,
            "timeout": 10,
            "timeout_short": 10,
            "timeout_long": 15,
            "max_citations": 5
        },
        "network": {
            "use_proxy": False,
            "log_network_requests": False,
            "http_proxy": "",
            "https_proxy": ""
        },
        "citation_extraction": {
            "use_llm_extraction": False,
            "use_llm_query_generation": False,
            "entity_extraction_model": "gpt-4",
            "query_generation_model": "gpt-4",
            "extraction_temperature": 0.1,
            "generation_temperature": 0.2,
            "extraction_max_tokens": 500,
            "generation_max_tokens": 500
        }
    }
    
    # Try to load from file, fall back to defaults if file doesn't exist
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with defaults (file config takes precedence)
            config = default_config.copy()
            for key, value in file_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
            
            return config
        else:
            # Return defaults if config file doesn't exist
            return default_config
            
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        return default_config

def load_environment_variables():
    """
    Load environment variables from .env file
    """
    load_dotenv()

import os
import streamlit as st
import json
from pathlib import Path

# Default database configuration
DEFAULT_DB_CONFIG = {
    "host": "clinexa.cgpek8igovya.us-east-1.rds.amazonaws.com",
    "port": 3306,
    "user": "clinexa",
    "password": "Am24268934",
    "database": "clinexa"
}

def get_db_config():
    """
    Get database configuration from various sources with priority:
    1. Streamlit secrets (for cloud deployment)
    2. Environment variables (for containerized deployment)
    3. .env file in project root (for local development)
    4. Default configuration (fallback)
    """
    # 1. Check Streamlit secrets (for Streamlit Cloud deployment)
    if hasattr(st, "secrets") and "mysql" in st.secrets:
        return {
            "host": st.secrets["connections"]["mysql"]["host"],
    "port": st.secrets["connections"]["mysql"]["port"],
    "user": st.secrets["connections"]["mysql"]["username"],
    "password": st.secrets["connections"]["mysql"]["password"],
    "database": st.secrets["connections"]["mysql"]["database"]
        }
    
    # 2. Check environment variables (for containerized deployment)
    env_config = {}
    if os.environ.get("DB_HOST"):
        env_config["host"] = os.environ.get("DB_HOST")
    if os.environ.get("DB_USER"):
        env_config["user"] = os.environ.get("DB_USER")
    if os.environ.get("DB_PASSWORD"):
        env_config["password"] = os.environ.get("DB_PASSWORD")
    if os.environ.get("DB_NAME"):
        env_config["database"] = os.environ.get("DB_NAME")
    
    if len(env_config) >= 4:  # If we have all required config from env vars
        return env_config
    
    # 3. Check .env file in project root
    try:
        dotenv_path = Path(__file__).parent.parent / ".env"
        if dotenv_path.exists():
            from dotenv import load_dotenv
            load_dotenv(dotenv_path)
            
            file_config = {}
            if os.environ.get("DB_HOST"):
                file_config["host"] = os.environ.get("DB_HOST")
            if os.environ.get("DB_USER"):
                file_config["user"] = os.environ.get("DB_USER")
            if os.environ.get("DB_PASSWORD"):
                file_config["password"] = os.environ.get("DB_PASSWORD")
            if os.environ.get("DB_NAME"):
                file_config["database"] = os.environ.get("DB_NAME")
            
            if len(file_config) >= 4:  # If we have all required config from .env
                return file_config
    except ImportError:
        # python-dotenv not installed
        pass
    
    # 4. Check for a config.json file
    try:
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_data = json.load(f)
                if "database" in config_data and len(config_data["database"]) >= 4:
                    return config_data["database"]
    except Exception:
        pass
    
    # 5. Fallback to default configuration
    return DEFAULT_DB_CONFIG

# Database connection parameters - use this in your files
DB_CONFIG = get_db_config() 

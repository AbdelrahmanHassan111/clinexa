import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mysql.connector
import datetime
import json
import time
import google.generativeai as genai
from streamlit_chat import message
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import os

# Gemini API setup
API_KEY = "AIzaSyDIST7Xvjns3VFMf2jbawPSX95cIhAkFhA"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Load the ML model - use relative path
model_path = "model/XGBoost_grid_optimized.joblib"
try:
    # Check if model file exists
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        st.warning(f"Model file not found at {model_path}. Please ensure the model file exists.")
        clf = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    clf = None

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "user": "root", 
    "password": "root",
    "database": "smart_clinic"
}

# Direct DB Connection
def get_db_connection():
    """Create a direct database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        return None 
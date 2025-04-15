# Hospital Management System

![Hospital Management System](https://img.shields.io/badge/Hospital%20Management%20System-Healthcare-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-purple)

A comprehensive hospital management system built with Streamlit, featuring patient management, doctor scheduling, and AI-powered disease prediction.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Key Components](#key-components)
- [Data Visualization](#data-visualization)
- [Security](#security)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### For Doctors
- **Patient Management**: Add, view, and manage patient information.
- **Alzheimer's Analysis**: Run machine learning-based predictions on patient data.
- **Medical Records**: Maintain comprehensive patient medical history.
- **AI Clinical Assistant**: Get AI-powered insights about patients.
- **Analytics Dashboard**: Visualize patient data and trends.

### For Administrators
- **User Management**: Create and manage doctor accounts.
- **Doctor Management**: Add and manage doctor profiles.
- **Patient Management**: Oversee all patient records.
- **Prediction Logs**: Monitor all Alzheimer's predictions.
- **Appointment Scheduling**: Manage patient appointments.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Database**: MySQL
- **Machine Learning**: XGBoost
- **AI Assistant**: Google Gemini API

## Project Structure

```
hospital_management_system/
├── app.py                   # Main application entry point
├── streamlit_app/           # Contains the Streamlit application code
│   ├── admin_view.py        # Admin interface
│   ├── doctor_view.py       # Doctor interface
│   ├── patient_view.py      # Patient interface
├── database/                # Database-related files
│   ├── db_creation.py       # Database initialization script
│   ├── db_creation.sql      # SQL schema
├── models/                  # Machine learning models
│   └── XGBoost_model.joblib  # Trained XGBoost model
├── utils/                   # Utility functions
└── requirements.txt         # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- MySQL 8.0 or higher
- Streamlit
- XGBoost
- Google Gemini API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/hospital-management-system.git
   cd hospital-management-system
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with the following variables:
   ```plaintext
   DB_HOST=your_database_host
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_NAME=your_database_name
   GOOGLE_API_KEY=your_google_api_key
   ```

5. **Initialize the database**:
   ```bash
   python database/db_creation.py
   ```

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application** at `http://localhost:8501`.

3. **Log in with the default credentials**:
   - Admin: username: `admin1`, password: `admin1`
   - Doctor: username: `dr.shaker`, password: `dr.shaker`

## Key Components

### Machine Learning Model

The application uses an XGBoost model trained on Alzheimer's disease data to predict patient outcomes. The model analyzes various clinical features including:

- Cognitive test scores (MMSE, CDRSB, ADAS13)
- Memory test results (RAVLT)
- Brain measurements (Hippocampus volume)
- Biomarkers (APOE4, TAU, ABETA)

### AI Clinical Assistant

The AI assistant uses Google's Gemini API to provide intelligent insights about patients. It can:

- Interpret test results
- Suggest treatment options
- Provide research-backed recommendations
- Answer questions about patient data

## Data Visualization

The application includes various visualizations:

- Feature importance plots
- Probability distribution charts
- Disease progression trends
- Patient analytics dashboards

## Security

- Role-based access control (Admin/Doctor)
- Secure password authentication
- Database connection security

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alzheimer's Disease Neuroimaging Initiative (ADNI) for data
- Google Gemini for AI capabilities
- Streamlit for the web framework
- XGBoost for the machine learning model
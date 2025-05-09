# Alzheimer's Diagnosis System

A comprehensive web application for Alzheimer's disease diagnosis that combines clinical data analysis, MRI scan processing, and AI-assisted diagnosis.

## Features

### For Doctors
- Patient management
- Alzheimer's analysis using machine learning models
- MRI scan processing with CNN and SWIN Transformer models
- Medical records management
- AI-assisted clinical consultation 
- Detailed scan interpretation and visualizations
- ROI measurements for brain structures

### For Patients (New)
- Online self-registration
- Appointment scheduling
- View medical records and test results
- Update personal information
- Access to Alzheimer's assessment results
- View MRI scan history

### For Administrators
- User management
- System monitoring
- Data analytics

## Tech Stack

- Frontend: Streamlit
- Backend: Python
- Database: MySQL
- AI Components: 
  - XGBoost model for clinical data analysis
  - CNN (ResNet50) and SWIN Transformer for MRI analysis
  - Google's Gemini Pro model for natural language generation

## System Requirements

- Python 3.9+
- MySQL 8.0+
- 8GB RAM or more (for running MRI analysis models)
- Internet connection (for AI assistant functionality)

## Quick Start Guide

1. **Install MySQL**
   - Make sure MySQL 8.0+ is installed and running
   - Create a database named `smart_clinic`
   
2. **Install Python dependencies**
   ```bash
   pip install streamlit mysql-connector-python pandas numpy matplotlib google-generativeai python-dotenv joblib scikit-learn
   ```

3. **Start the application**
   ```bash
   cd streamlit_app
   streamlit run app.py
   ```

4. **Access the web interface**
   - Open your browser and go to: http://localhost:8501

## Database Setup

If you need to manually update the database schema:

```bash
cd streamlit_app
python -c "from deploy import update_database_schema; update_database_schema()"
```

## Usage

### Doctor Portal

Log in using your doctor credentials to:
- Manage patients
- Perform Alzheimer's analyses
- Process MRI scans
- View and update medical records
- Use AI-assisted consultation

### Patient Portal

1. Click on the "Patient Portal" button on the login page
2. Register a new account or log in with existing credentials
3. Schedule new appointments
4. View upcoming and past appointments
5. Access medical records and test results
6. Update personal information

### Admin Portal

Log in using admin credentials to:
- Manage user accounts
- Monitor system usage
- View analytics

## Online Deployment

The system can be deployed online using:

1. Streamlit Sharing: https://streamlit.io/sharing
2. Docker container with Docker Compose
3. Cloud services like AWS, Azure, or Google Cloud

For online deployment, make sure to:
- Configure a secure database connection
- Set up proper authentication
- Configure environment variables for sensitive information

## Troubleshooting

**MySQL Connection Issues**
- Verify MySQL is running
- Check connection parameters in DB_CONFIG within app.py
- Ensure the smart_clinic database exists

**Missing Dependencies**
- Run `pip install -r requirements.txt` to install all dependencies

**Database Schema Issues**
- Run the database update script manually:
  ```python
  from deploy import update_database_schema
  update_database_schema()
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact the development team at support@example.com
# Hospital Management System

A comprehensive hospital management system built with Streamlit, featuring patient management, doctor scheduling, and AI-powered disease prediction.

## Features

- **Patient Management**: Register and manage patient information
- **Doctor Scheduling**: Schedule appointments and manage doctor availability
- **AI Disease Prediction**: Predict disease progression using machine learning
- **Admin Dashboard**: Comprehensive admin panel for system management
- **Doctor Interface**: Dedicated interface for doctors to manage patients and appointments

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hospital-management-system.git
   cd hospital-management-system
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   DB_HOST=your_database_host
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_NAME=your_database_name
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Initialize the database:
   ```
   python database/db_creation.py
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Access the application at `http://localhost:8501`

## Project Structure

- `app.py`: Main application entry point
- `streamlit_app/`: Contains the Streamlit application code
  - `admin_view.py`: Admin interface
  - `doctor_view.py`: Doctor interface
  - `patient_view.py`: Patient interface
- `database/`: Database-related files
  - `db_creation.py`: Database initialization script
  - `db_creation.sql`: SQL schema
- `models/`: Machine learning models
- `utils/`: Utility functions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
# Enhanced Patient Portal

This is an enhanced UI implementation of the patient portal for the Alzheimer's Diagnosis System. It provides a modern, user-friendly interface for patients to access their medical records, schedule appointments, view MRI scan results, and manage their profiles.

## Features

- **Modern UI**: Clean, responsive design with intuitive navigation
- **Interactive Dashboard**: View health metrics, upcoming appointments, and personalized recommendations
- **Medical Records**: Access and search through clinical records, brain scans, and cognitive assessments with visualizations
- **Appointment Scheduling**: Schedule, manage, and cancel appointments with doctors
- **Profile Management**: View and update personal information, account settings, and notification preferences

## Directory Structure

```
patient_portal/
├── __init__.py            # Package initialization
├── main.py                # Main entry point
├── components/           # Reusable UI components
│   ├── __init__.py
│   └── navigation.py     # Sidebar navigation component
├── pages/                # Individual page modules
│   ├── __init__.py
│   ├── auth.py           # Login and registration pages
│   ├── dashboard.py      # Main dashboard page
│   ├── appointments.py   # Appointment scheduling and management
│   ├── medical_records.py # Medical records and test results
│   └── profile.py        # User profile and settings
├── utils/                # Utility functions and helpers
│   ├── __init__.py
│   └── db.py             # Database connection and queries
└── static/               # Static assets
    └── style.css         # Custom CSS styling
```

## Integration

The enhanced patient portal is designed to integrate with the existing Alzheimer's Diagnosis System. It can be used in two ways:

1. **Standalone Mode**: For development and testing using `run_enhanced_portal.py`
2. **Integrated Mode**: As part of the main application via the updated `app.py`

## Running the Application

### Standalone Mode (for development)

```bash
# From the project root directory
streamlit run run_enhanced_portal.py
```

### Integrated Mode (with the full application)

```bash
# From the project root directory
cd streamlit_app
streamlit run app.py
```

Then log in with patient credentials or use the demo mode option.

## Dependencies

- Streamlit
- Pandas
- Plotly
- MySQL Connector
- Datetime

## Design Choices

- **Modular Structure**: Separated components, pages, and utilities for better organization and maintainability
- **Modern UI Components**: Cards, metrics, and visualizations for better data presentation
- **Responsive Design**: Works well on desktop and mobile devices
- **Consistent Styling**: Unified color scheme and component design across all pages

## Future Enhancements

- Real-time notifications for test results and appointment confirmations
- Integration with telehealth services for virtual consultations
- Enhanced data visualizations for cognitive health trends
- Mobile app version using the same design principles 
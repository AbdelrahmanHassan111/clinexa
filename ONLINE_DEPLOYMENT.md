# Alzheimer's Diagnosis System - Online Deployment Guide

This guide provides step-by-step instructions for deploying your Alzheimer's Diagnosis System online, allowing patients to register and schedule appointments from anywhere in the world.

## Quick Start

For Windows users, the simplest way to deploy your application online is:

1. Double-click the `deploy_online.bat` file in your project root directory
2. Follow the on-screen instructions

## Deployment Options

### Option 1: Temporary URL with ngrok (Easiest)

This method creates a tunnel to your locally running application, making it available through a public URL:

1. Make sure Python and all dependencies are installed
2. Run the deployment script:
   ```
   cd streamlit_app
   python deploy_online.py
   ```
3. Share the provided URL with your patients

**Limitations:**
- The URL changes each time you restart ngrok
- Your computer must stay on with the script running
- Free ngrok accounts have limited bandwidth

### Option 2: Streamlit Cloud (Recommended for Production)

For a permanent solution with a fixed URL:

1. Create a GitHub repository for your project
2. Push your code to the repository
3. Sign up for Streamlit Cloud (https://streamlit.io/cloud)
4. Deploy your application through Streamlit Cloud
5. Configure secrets for database access

**Advantages:**
- Permanent URL that never changes
- Always online (no need to keep your computer running)
- Professionally hosted service

## Database Configuration for Online Deployment

When deploying online, you'll need to consider how to connect to your database:

### For Option 1 (ngrok):
- Your database must be accessible to your local machine
- The default configuration will work if MySQL is running locally

### For Option 2 (Streamlit Cloud):
1. Set up a cloud-hosted MySQL database (AWS RDS, Google Cloud SQL, etc.)
2. Configure Streamlit secrets by creating a file `.streamlit/secrets.toml` with:
   ```toml
   [mysql]
   host = "your-cloud-db-host"
   user = "your-db-username"
   password = "your-db-password"
   database = "smart_clinic"
   ```

## Troubleshooting

### Connection Issues
- **Cannot access URL**: Make sure your firewall isn't blocking ngrok
- **Database connection fails**: Check your database configuration
- **Application crashes**: Check logs and ensure all dependencies are installed

### Database Issues
- **Cannot connect to remote database**: Ensure your database allows remote connections
- **Access denied**: Check your database credentials

## Support

If you need help with deployment, please contact support@example.com 
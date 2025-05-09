# Online Deployment Guide

This guide provides instructions for deploying your Alzheimer's Diagnosis System online so patients can access it from anywhere in the world.

## Option 1: Quick Deployment with ngrok (Temporary URL)

This method creates a secure tunnel to your locally running application, making it accessible through a public URL immediately.

### Prerequisites
- Your application running locally (on your laptop)
- Internet connection

### Setup Steps

1. **Install ngrok**

   ```powershell
   # Using PowerShell with admin privileges:
   choco install ngrok
   ```
   
   If you don't have Chocolatey installed, you can download ngrok directly from [ngrok.com](https://ngrok.com/download) and extract it to a folder on your computer.

2. **Sign up for a free ngrok account**
   - Go to [https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)
   - After signing up, find your authtoken in the dashboard

3. **Connect your ngrok account**

   ```powershell
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   ```

4. **Start your Streamlit application**

   ```powershell
   cd "D:\Semester 6\database_project\streamlit_app"
   streamlit run app.py
   ```

5. **Create the tunnel with ngrok**
   
   In a new PowerShell window:
   ```powershell
   ngrok http 8501
   ```

6. **Share the URL**
   - The ngrok terminal will display a URL like `https://1234-abcd-5678.ngrok.io`
   - Share this URL with your patients to access the system from anywhere

### Important Notes
- The free ngrok plan provides URLs that change each time you restart ngrok
- The tunnel remains active only as long as your computer is running both the Streamlit app and ngrok
- Make sure your firewall isn't blocking the connection

## Option 2: Permanent Cloud Deployment

For a permanent solution with a fixed URL, you can deploy to Streamlit Cloud.

### Prerequisites
- GitHub account
- Your project code in a GitHub repository

### Setup Steps

1. **Create a GitHub repository**
   - Create a new repository on GitHub
   - Push your project code to this repository

2. **Create a requirements.txt file**
   Create a file named `requirements.txt` in your project root with the following content:

   ```
   streamlit>=1.30.0
   pandas>=2.1.0
   numpy>=1.25
   mysql-connector-python==8.0.33
   xgboost==1.7.5
   scikit-learn==1.2.2
   matplotlib==3.7.1
   seaborn==0.12.2
   google-generativeai==0.3.1
   python-dotenv==1.0.0
   joblib==1.2.0
   ```

3. **Sign up for Streamlit Cloud**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign up using your GitHub account

4. **Deploy your application**
   - In Streamlit Cloud dashboard, click "New app"
   - Select your GitHub repository
   - Enter the path to your main app file: `streamlit_app/app.py`
   - Click "Deploy"

5. **Set up database access**
   For this to work with your MySQL database, you'll need to:
   - Use a cloud-hosted MySQL database (like AWS RDS, Google Cloud SQL, or a MySQL hosting service)
   - Update the DB_CONFIG in the deployed app using Streamlit's secrets management

6. **Share the permanent URL**
   - Once deployed, Streamlit provides a fixed URL for your application (like `https://your-app-name.streamlit.app`)
   - This URL can be shared with patients and will remain consistent

## Option 3: Deploy on a VPS (Advanced)

For complete control over your deployment, you can use a Virtual Private Server (VPS) like DigitalOcean, AWS EC2, or Google Compute Engine.

1. **Set up a VPS** with Ubuntu
2. **Install dependencies** (Python, MySQL)
3. **Clone your repository** to the server
4. **Install project requirements**
5. **Configure MySQL** on the server
6. **Set up NGINX** as a reverse proxy
7. **Configure SSL** with Let's Encrypt
8. **Create a systemd service** to keep your app running

This option requires more technical expertise but gives you full control over your deployment.

## Choosing the Right Option

- **Option 1 (ngrok)**: Best for quick testing, demos, or temporary access
- **Option 2 (Streamlit Cloud)**: Best for permanent deployment with minimal configuration
- **Option 3 (VPS)**: Best for production use with full control over your environment 
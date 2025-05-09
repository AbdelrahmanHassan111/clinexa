# Online Deployment Guide

## Share Your Alzheimer's Diagnosis System with Patients Anywhere

This guide provides simple steps to make your system accessible online so patients can register and schedule appointments from anywhere in the world.

## Quick Start (One-Click Solution)

1. Double-click the `START_ONLINE.bat` file in your project root directory
2. When prompted, enter your ngrok authtoken (see below for how to get one)
3. Share the displayed URL with your patients

## Getting an ngrok authtoken

Before you can deploy your application online, you need an ngrok authtoken:

1. Sign up for a free account at [ngrok.com/signup](https://dashboard.ngrok.com/signup)
2. After signing in, go to the [Auth page](https://dashboard.ngrok.com/get-started/your-authtoken)
3. Copy your authtoken - it looks something like `2ABCdefGhIJklMNOpqrsTU1v2w3X_4YzaBCDEFghiJk`

## What's Happening Behind the Scenes

When you run the deployment script:

1. Your Streamlit application starts on your computer
2. ngrok creates a secure tunnel to your local application
3. A public URL is generated that anyone can use to access your system
4. This URL is displayed in the terminal and saved to `current_url.txt`

## Important Notes

- **The URL is temporary**: It will change each time you restart the script
- **Your computer must stay running**: The application is hosted on your computer
- **Free tier limitations**: The free ngrok plan has usage limitations:
  - 1 active tunnel at a time
  - 40 connections per minute
  - Online for as long as the script is running

## Using a Custom Domain (Optional)

With a free ngrok account, you can create a custom subdomain to use:

1. Log in to your [ngrok dashboard](https://dashboard.ngrok.com/)
2. Go to Domains → Create a Domain
3. Choose a name for your subdomain (e.g., `your-clinic.ngrok.io`)
4. Edit the `start_online.ps1` script to add the `-subdomain your-clinic` parameter to the ngrok command

## For Advanced Users: Cloud Deployment

For a permanent solution with 24/7 availability, consider deploying to a cloud provider:

- **Streamlit Cloud**: Easiest option. Push your code to GitHub and connect to [Streamlit Cloud](https://streamlit.io/cloud)
- **Heroku**: Good free tier with easy deployment
- **Microsoft Azure**: Robust solution with free credits for startups
- **DigitalOcean**: Simple cloud VPS with fixed pricing

For more details on cloud deployment, please see our separate Cloud Deployment Guide.

## Troubleshooting

### "ngrok not found"
- Make sure `ngrok.exe` is in your project root directory
- You can download it from [ngrok.com/download](https://ngrok.com/download)

### "Streamlit not found"
- Make sure Streamlit is installed: `pip install streamlit streamlit-chat`

### "Authentication error"
- Double-check your authtoken
- Run `ngrok authtoken YOUR_TOKEN` manually

### "Address already in use"
- Another application is using port 8501
- Stop other Streamlit instances or change the port

### "Application error"
- Check if MySQL is running
- Verify database connection settings

## Getting Help

If you encounter any issues, please check:
1. The Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io/)
2. The ngrok documentation: [ngrok.com/docs](https://ngrok.com/docs)
3. Our project documentation in the README.md file

Happy deploying! Your patients will now be able to access your system from anywhere in the world. 
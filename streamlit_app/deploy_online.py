import os
import subprocess
import sys
import time
import webbrowser
import platform
import requests
import json
import streamlit as st
from pathlib import Path

def check_ngrok_installed():
    """Check if ngrok is installed and available in PATH or current directory"""
    try:
        # First check if ngrok.exe exists in current directory
        if os.path.exists("ngrok.exe"):
            return True
            
        # Otherwise check in PATH
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ngrok"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "ngrok"], capture_output=True, text=True)
        
        return result.returncode == 0
    except Exception:
        return False

def get_ngrok_cmd():
    """Get the appropriate ngrok command to use"""
    if os.path.exists("ngrok.exe"):
        return ".\\ngrok.exe"
    else:
        return "ngrok"

def install_ngrok():
    """Guide the user to install ngrok"""
    print("\n" + "="*80)
    print("NGROK NOT FOUND")
    print("="*80)
    print("\nTo deploy your application online, you need to install ngrok.")
    
    if platform.system() == "Windows":
        print("\nOption 1: Install with Chocolatey (if installed):")
        print("    Run in admin PowerShell: choco install ngrok")
        
        print("\nOption 2: Download and install manually:")
        print("    1. Go to https://ngrok.com/download")
        print("    2. Download the Windows version")
        print("    3. Extract the zip file")
        print("    4. Add the folder containing ngrok.exe to your PATH environment variable")
        print("       or copy ngrok.exe to your project directory")
    
    elif platform.system() == "Darwin":  # macOS
        print("\nOption 1: Install with Homebrew:")
        print("    brew install ngrok")
        
        print("\nOption 2: Download and install manually:")
        print("    1. Go to https://ngrok.com/download")
        print("    2. Download the macOS version")
        print("    3. Extract the zip file")
        print("    4. Move ngrok to /usr/local/bin")
    
    else:  # Linux
        print("\nOption 1: Install with package manager:")
        print("    sudo snap install ngrok    # If you use Snap")
        
        print("\nOption 2: Download and install manually:")
        print("    1. Go to https://ngrok.com/download")
        print("    2. Download the Linux version")
        print("    3. Extract the zip file")
        print("    4. Move ngrok to /usr/local/bin")
    
    print("\nAfter installing ngrok, run this script again.")
    
    return False

def check_ngrok_auth():
    """Check if ngrok is authenticated"""
    try:
        # Run ngrok config check to see if it returns any issues
        ngrok_cmd = get_ngrok_cmd()
        result = subprocess.run([ngrok_cmd, "config", "check"], capture_output=True, text=True)
        if "error" in result.stdout.lower() or "not logged in" in result.stdout.lower():
            return False
        return True
    except Exception:
        return False

def setup_ngrok_auth():
    """Guide the user to set up ngrok authentication"""
    print("\n" + "="*80)
    print("NGROK AUTHENTICATION REQUIRED")
    print("="*80)
    print("\nTo use ngrok, you need to create an account and connect your authtoken.")
    print("\nStep 1: Create a free ngrok account at https://dashboard.ngrok.com/signup")
    print("Step 2: Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken")
    
    authtoken = input("\nEnter your ngrok authtoken: ").strip()
    
    if not authtoken:
        print("No authtoken provided. Please sign up and get your token from ngrok.")
        return False
    
    try:
        # Add the authtoken to ngrok config
        ngrok_cmd = get_ngrok_cmd()
        subprocess.run([ngrok_cmd, "config", "add-authtoken", authtoken], check=True)
        print("\nNgrok authentication successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError setting up ngrok authentication: {e}")
        return False

def start_streamlit_app():
    """Start the Streamlit application"""
    print("\nStarting Streamlit application...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    # Give Streamlit time to start
    time.sleep(5)
    return streamlit_process

def start_ngrok_tunnel(port=8501):
    """Start an ngrok tunnel to the specified port"""
    print("\nStarting ngrok tunnel...")
    ngrok_cmd = get_ngrok_cmd()
    ngrok_process = subprocess.Popen(
        [ngrok_cmd, "http", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give ngrok time to start
    time.sleep(3)
    
    # Get the public URL from ngrok API
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        data = response.json()
        
        if "tunnels" in data and len(data["tunnels"]) > 0:
            public_url = data["tunnels"][0]["public_url"]
            
            # Replace http with https if needed
            if public_url.startswith("http://"):
                public_url = "https://" + public_url[7:]
                
            print("\n" + "="*80)
            print(f"🎉 YOUR APPLICATION IS NOW ONLINE! 🎉")
            print("="*80)
            print(f"\nPublic URL: {public_url}")
            print("\nShare this URL with your patients so they can access your system from anywhere!")
            print("\nIMPORTANT NOTES:")
            print("1. This URL will change if you restart ngrok")
            print("2. Your computer must stay on with this script running")
            print("3. For a permanent solution, follow the cloud deployment guide")
            print("\nPress Ctrl+C to stop the server when you're done")
            
            # Write the URL to a file for reference
            with open("online_url.txt", "w") as f:
                f.write(f"Your application is available at: {public_url}\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\nNote: This URL will change if you restart ngrok.\n")
            
            # Try to open the URL in the default browser
            try:
                webbrowser.open(public_url)
            except:
                pass
                
            return ngrok_process, public_url
    except Exception as e:
        print(f"Error getting ngrok public URL: {e}")
    
    print("Failed to get ngrok public URL. Check if ngrok is running correctly.")
    return ngrok_process, None

def main():
    print("\n" + "="*80)
    print("ALZHEIMER'S DIAGNOSIS SYSTEM - ONLINE DEPLOYMENT")
    print("="*80)
    
    # Check if ngrok is installed
    if not check_ngrok_installed():
        install_ngrok()
        return
    
    # Check if ngrok is authenticated
    if not check_ngrok_auth():
        if not setup_ngrok_auth():
            return
    
    # Start the Streamlit app
    streamlit_process = start_streamlit_app()
    
    # Start the ngrok tunnel
    ngrok_process, public_url = start_ngrok_tunnel()
    
    if not public_url:
        print("Failed to deploy the application online.")
        streamlit_process.terminate()
        if ngrok_process:
            ngrok_process.terminate()
        return
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        streamlit_process.terminate()
        ngrok_process.terminate()

if __name__ == "__main__":
    main() 
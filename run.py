#!/usr/bin/env python3
"""
Quick Start Script for Alpha Recommendation Engine
Checks dependencies and launches the Streamlit dashboard
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is 3.9 or higher"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_package(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def main():
    print("=" * 80)
    print("🚀 ALPHA RECOMMENDATION ENGINE - QUICK START")
    print("=" * 80)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check critical packages
    critical_packages = ['streamlit', 'pandas', 'numpy', 'yfinance', 'plotly']
    missing_packages = [pkg for pkg in critical_packages if not check_package(pkg)]
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("\n❌ Cannot proceed without required packages")
            print("   Run: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✅ All critical packages are installed")
    
    # Launch Streamlit
    print("\n" + "=" * 80)
    print("🎨 Launching Streamlit Dashboard...")
    print("=" * 80)
    print("\n📍 Dashboard will open at: http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard_app.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped successfully")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

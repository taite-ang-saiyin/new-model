#!/usr/bin/env python3
"""
CogniVerse Setup Script
Installs and configures all three systems
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_requirements():
    """Install Python requirements"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python requirements",
    )


def download_spacy_model():
    """Download spaCy English model"""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model",
    )


def download_nltk_data():
    """Download NLTK data"""
    nltk_script = """
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data downloaded successfully")
"""

    return run_command(f'{sys.executable} -c "{nltk_script}"', "Downloading NLTK data")


def create_directories():
    """Create necessary directories"""
    directories = ["logs", "cache", "results", "models"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_sample_config():
    """Create sample configuration"""
    from config import create_sample_config

    create_sample_config()


def verify_installation():
    """Verify that all systems can be imported"""
    print("🔍 Verifying installation...")

    # Test Model Architect
    try:
        sys.path.append("model_architect")
        from optimized_router import OptimizedRouter

        print("✅ Model Architect: OptimizedRouter import successful")
    except Exception as e:
        print(f"⚠️ Model Architect: {e}")

    # Test Narrative Weaver
    try:
        sys.path.append("narrative_weaver/src")
        from narrative_weaver.story_generator import get_generator

        print("✅ Narrative Weaver: Story generator import successful")
    except Exception as e:
        print(f"⚠️ Narrative Weaver: {e}")

    # Test Action Engine
    try:
        sys.path.append("action_engine")
        from models.policy_networks import PolicyNetwork

        print("✅ Action Engine: Policy networks import successful")
    except Exception as e:
        print(f"⚠️ Action Engine: {e}")

    # Test main bridge
    try:
        from CogniVerse_API_Bridge import get_cogniverse_bridge

        bridge = get_cogniverse_bridge()
        status = bridge.get_system_status()
        print("✅ CogniVerse Bridge: Initialization successful")
        print(f"   Systems available: {sum(status.values())}/{len(status)}")
    except Exception as e:
        print(f"❌ CogniVerse Bridge: {e}")


def main():
    """Main setup function"""
    print("🚀 CogniVerse Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        sys.exit(1)

    # Download language models
    download_spacy_model()
    download_nltk_data()

    # Create directories
    create_directories()

    # Create sample config
    create_sample_config()

    # Verify installation
    verify_installation()

    print("\n🎉 CogniVerse setup completed!")
    print("\nNext steps:")
    print("1. Run: python main.py interactive")
    print("2. Or run: python main.py test")
    print("3. Or use the API bridge directly: python CogniVerse_API_Bridge.py --help")


if __name__ == "__main__":
    main()

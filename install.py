#!/usr/bin/env python3
"""
Enhanced RAG Pipeline System - Automated Installation Script
Installs all dependencies, downloads models, and sets up the system
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description="", check=True):
    """Run a command with error handling"""
    print(f"🔄 {description}")
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   ❌ Details: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def detect_gpu():
    """Detect if CUDA GPU is available"""
    print("🖥️  Detecting GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA GPU detected: {gpu_name}")
            return True
        else:
            print("   ℹ️  No CUDA GPU detected, will use CPU")
            return False
    except ImportError:
        print("   ℹ️  PyTorch not installed yet, will detect GPU after installation")
        return False

def install_pytorch(use_gpu=False):
    """Install PyTorch with appropriate backend"""
    print("🔥 Installing PyTorch...")
    
    if use_gpu:
        # Install GPU version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        description = "Installing PyTorch with CUDA support"
    else:
        # Install CPU version
        command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        description = "Installing PyTorch (CPU version)"
    
    return run_command(command, description)

def install_faiss(use_gpu=False):
    """Install FAISS with appropriate backend"""
    print("🔍 Installing FAISS...")
    
    if use_gpu:
        command = "pip install faiss-gpu"
        description = "Installing FAISS with GPU support"
    else:
        command = "pip install faiss-cpu"
        description = "Installing FAISS (CPU version)"
    
    return run_command(command, description)

def install_core_dependencies():
    """Install core ML/NLP dependencies"""
    print("📦 Installing core dependencies...")
    
    dependencies = [
        "transformers>=4.20.0",
        "sentence-transformers>=2.2.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "spacy>=3.4.0",
        "nltk>=3.7",
        "regex>=2022.7.9",
        "datasets>=2.0.0",
        "networkx>=2.8.0",
        "tqdm>=4.64.0",
        "requests>=2.28.0",
        "python-dateutil>=2.8.0",
        "typing-extensions>=4.0.0",
        "jsonschema>=4.4.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
        "asyncio-throttle>=1.0.2",
        "collections-extended>=2.0.2"
    ]
    
    for dep in dependencies:
        success = run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}")
        if not success:
            print(f"   ⚠️  Failed to install {dep}, continuing...")

def install_development_dependencies():
    """Install development and testing dependencies"""
    print("🧪 Installing development dependencies...")
    
    dev_dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.18.0",
        "mock>=4.0.3"
    ]
    
    for dep in dev_dependencies:
        run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}", check=False)

def download_spacy_model():
    """Download spaCy English model"""
    print("🌐 Downloading spaCy English model...")
    return run_command("python -m spacy download en_core_web_sm", "Downloading en_core_web_sm")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    nltk_commands = [
        "import nltk; nltk.download('punkt')",
        "import nltk; nltk.download('stopwords')",
        "import nltk; nltk.download('wordnet')"
    ]
    
    for cmd in nltk_commands:
        run_command(f'python -c "{cmd}"', f"Downloading NLTK data", check=False)

def download_tinyllama():
    """Pre-download TinyLlama model"""
    print("🤖 Pre-downloading TinyLlama model (~2.2GB)...")
    
    download_script = '''
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Downloading TinyLlama tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("Downloading TinyLlama model...")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("TinyLlama download completed!")
'''
    
    return run_command(f'python -c "{download_script}"', "Downloading TinyLlama model", check=False)

def verify_installation():
    """Verify that key components can be imported"""
    print("✅ Verifying installation...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("faiss", "FAISS"),
        ("spacy", "spaCy"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    success_count = 0
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name} - OK")
            success_count += 1
        except ImportError:
            print(f"   ❌ {name} - Failed to import")
    
    print(f"\n📊 Installation verification: {success_count}/{len(test_imports)} components working")
    return success_count == len(test_imports)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "pipeline_output",
        "pipeline_output/embeddings",
        "models_cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}")

def main():
    """Main installation process"""
    print("🚀 ENHANCED RAG PIPELINE SYSTEM - AUTOMATED INSTALLATION")
    print("=" * 80)
    print("This script will install all dependencies and download required models.")
    print("Estimated download size: ~3-4GB (models + dependencies)")
    print("Estimated time: 10-20 minutes (depending on internet speed)")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Detect GPU
    has_gpu = detect_gpu()
    
    # Ask user about GPU usage
    if has_gpu:
        use_gpu = input("\n🖥️  GPU detected. Install GPU-accelerated versions? (y/n): ").lower().startswith('y')
    else:
        use_gpu = False
        print("\n💻 Installing CPU-only versions")
    
    print(f"\n🎯 Installation mode: {'GPU-accelerated' if use_gpu else 'CPU-only'}")
    
    # Confirm installation
    confirm = input("\n▶️  Continue with installation? (y/n): ").lower().startswith('y')
    if not confirm:
        print("Installation cancelled.")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("STARTING INSTALLATION")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Install PyTorch first (required by many other packages)
    if not install_pytorch(use_gpu):
        print("❌ Failed to install PyTorch. Aborting installation.")
        sys.exit(1)
    
    # Install FAISS
    if not install_faiss(use_gpu):
        print("❌ Failed to install FAISS. Aborting installation.")
        sys.exit(1)
    
    # Install core dependencies
    install_core_dependencies()
    
    # Install development dependencies
    install_development_dependencies()
    
    # Download models and data
    download_spacy_model()
    download_nltk_data()
    
    # Pre-download TinyLlama (optional, may take time)
    download_tinyllama_choice = input("\n🤖 Pre-download TinyLlama model now? (~2.2GB, recommended) (y/n): ").lower().startswith('y')
    if download_tinyllama_choice:
        download_tinyllama()
    else:
        print("   ℹ️  TinyLlama will be downloaded automatically on first use")
    
    # Verify installation
    print("\n" + "=" * 80)
    print("VERIFYING INSTALLATION")
    print("=" * 80)
    
    if verify_installation():
        print("\n🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("✅ All dependencies installed")
        print("✅ Models downloaded")
        print("✅ System ready to use")
        print("\n📖 Next steps:")
        print("1. Run data ingestion: python Ingestion_pipeline.py")
        print("2. Test complete pipeline: python test_complete_pipeline.py")
        print("\n🚀 Happy RAG Pipeline Building!")
    else:
        print("\n⚠️  INSTALLATION COMPLETED WITH WARNINGS")
        print("=" * 80)
        print("Some components may not work properly.")
        print("Check the error messages above and try manual installation.")
        print("\n🔧 Manual installation commands:")
        print("pip install torch transformers sentence-transformers faiss-cpu spacy")
        print("python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()
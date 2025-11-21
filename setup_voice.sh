#!/bin/bash

echo "🚀 Setting up Voice-Enabled RAG Chatbot"
echo "========================================="

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies for audio (macOS)
echo "🔊 Installing audio system dependencies..."

# Install ffmpeg for audio processing
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing ffmpeg..."
    brew install ffmpeg
else
    echo "✅ ffmpeg already installed"
fi

# Install portaudio for audio recording
if ! command -v portaudio &> /dev/null; then
    echo "Installing portaudio..."
    brew install portaudio
else
    echo "✅ portaudio already installed"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set up Google Cloud credentials"
echo "2. Add PDFs to data/pdf/"
echo "3. Run: python app.py"
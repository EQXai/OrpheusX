#!/bin/bash
set -e  # Exit on error

echo "======================================="
echo "OrpheusTTS-WebUI Setup Script"
echo "======================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip

# Install other required packages
echo "Installing other dependencies..."
pip install orpheus-speech gradio vllm torch huggingface_hub

# Create launch script
echo "Creating launch script..."
cat > launch_orpheus.sh << 'EOF'
#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

# Execute the wrapper script
python orpheus_wrapper.py
EOF

# Make the launch script executable
chmod +x launch_orpheus.sh

# Log in to Hugging Face
echo "======================================="
echo "You need to log in to Hugging Face to access the model."
echo "If you don't have an account, create one at https://huggingface.co/join"
echo "======================================="
read -p "Press Enter to continue to Hugging Face login..."
huggingface-cli login

# Remind about model access
echo "======================================="
echo "IMPORTANT: The OrpheusTTS model is a gated model."
echo "You need to request access at:"
echo "https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod"
echo "https://huggingface.co/canopylabs/orpheus-3b-0.1-pretrained"
echo "======================================="
echo "Once approved, you'll be able to use the model."
echo "======================================="

# Make the wrapper executable
echo "Making the wrapper script executable..."
chmod +x orpheus_wrapper.py

echo "Setup complete! Run ./launch_orpheus.sh to start the application."
echo "=======================================" 

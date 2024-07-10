#!/bin/bash

set -e

echo "Starting automated setup for FloatStream.io with Streamlit frontend..."

# Function to detect OS and package manager
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "debian"
        elif command -v yum &> /dev/null; then
            echo "fedora"
        else
            echo "unknown_linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Detect the OS
OS=$(detect_os)

# Function to install packages based on OS
install_packages() {
    case $OS in
        debian)
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv curl
            ;;
        fedora)
            sudo yum update -y
            sudo yum install -y python3 python3-pip python3-virtualenv curl
            ;;
        macos)
            if ! command -v brew &> /dev/null; then
                echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
                echo "After installing Homebrew, run this script again."
                exit 1
            fi
            brew update
            brew install python@3 curl
            ;;
        *)
            echo "Unsupported operating system. Please install Python 3, pip, and curl manually."
            exit 1
            ;;
    esac
}

# Install necessary packages
echo "Installing system dependencies..."
install_packages

# Cleanup section: Remove all saved files and data
echo "Cleaning up old data and files..."
rm -rf data
rm -f faiss_hnsw_index.bin
rm -f documents.pkl

# Install Ollama
echo "Installing Ollama..."
if [[ "$OS" == "macos" ]]; then
    brew install ollama
else
    curl https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "Starting Ollama service..."
if [[ "$OS" != "macos" ]]; then
    if command -v systemctl &> /dev/null; then
        if systemctl is-active --quiet ollama; then
            echo "Ollama service is already running."
        else
            echo "Attempting to start Ollama service. You may be prompted for your password."
            sudo systemctl start ollama
        fi
    else
        echo "Unable to start Ollama service automatically. Please start it manually."
    fi
else
    echo "On macOS, please start Ollama manually if it's not already running."
    echo "You can typically do this by running 'ollama serve' in a separate terminal."
fi

# Create a virtual environment
echo "Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Check if requirements.txt exists, if not, create it
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found. Creating it..."
    cat > requirements.txt << EOL
fastapi
uvicorn
pydantic
langchain
numpy
faiss-cpu
ollama
PyPDF2
python-multipart
chardet
pandas
plotly
streamlit
streamlit-mermaid
requests
python-dotenv
sqlparse
rank_bm25
EOL
    echo "requirements.txt created."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data

# Check if config.json exists, if not, create a sample one
if [ ! -f "config.json" ]; then
    echo "config.json not found. Creating a sample configuration..."
    cat > config.json << EOL
{
    "model_name": "qwen2",
    "embedding_model_name": "mxbai-embed-large",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "search_k": 20,
    "ef_search": 100,
    "index_file": "faiss_hnsw_index.bin",
    "docs_file": "documents.pkl"
}
EOL
    echo "Sample config.json created. Please review and adjust as needed."
fi

# Copy configuration file
echo "Copying configuration file..."
cp config.json data/config.json

# Download the Ollama model
echo "Downloading the Ollama model (this may take a while)..."
ollama pull qwen2
ollama pull mxbai-embed-large

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please ensure it's in the current directory."
    exit 1
fi

# Check if frontend.py exists
if [ ! -f "frontend.py" ]; then
    echo "Error: frontend.py not found. Please ensure it's in the current directory."
    exit 1
fi

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to kill process using a specific port
kill_process_on_port() {
    local pid=$(lsof -t -i :$1)
    if [ ! -z "$pid" ]; then
        echo "Killing process using port $1"
        kill -9 $pid
    fi
}

# Check and kill processes on ports 8000 and 8501 if they're in use
if port_in_use 8000; then
    echo "Port 8000 is already in use. Attempting to kill the process..."
    kill_process_on_port 8000
    sleep 2
fi

if port_in_use 8501; then
    echo "Port 8501 is already in use. Attempting to kill the process..."
    kill_process_on_port 8501
    sleep 2
fi

# Run the FastAPI backend
echo "Starting the FastAPI backend..."
python3 app.py &
BACKEND_PID=$!

# Wait for the backend to start
echo "Waiting for the backend to start..."
max_retries=30
retry_count=0
while ! curl -s http://localhost:8000/health > /dev/null
do
    sleep 1
    retry_count=$((retry_count+1))
    if [ $retry_count -ge $max_retries ]; then
        echo "Backend failed to start after $max_retries attempts. Exiting..."
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done
echo "Backend started successfully."

# Run the Streamlit frontend
echo "Starting the Streamlit frontend..."
streamlit run frontend.py &
FRONTEND_PID=$!

echo "Application is running."
echo "Access the Streamlit frontend at http://localhost:8501"
echo "The FastAPI backend is running at http://localhost:8000"
echo "Press Ctrl+C to stop the application."

# Wait for processes to finish
wait $BACKEND_PID $FRONTEND_PID

# Clean up
echo "Application finished. Cleaning up..."
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
if [[ "$OS" != "macos" ]]; then
    if command -v systemctl &> /dev/null; then
        sudo systemctl stop ollama
    fi
fi

echo "Setup and execution complete."

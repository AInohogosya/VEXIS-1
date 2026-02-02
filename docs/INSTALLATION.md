# Installation Guide

This guide covers installation and setup of the AI Agent system across different platforms.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space
- Administrative privileges (for GUI automation)

### Platform-Specific Requirements

#### macOS
- macOS 10.15 (Catalina) or later
- Xcode Command Line Tools
- Accessibility permissions for GUI automation

#### Windows
- Windows 10 or later
- Microsoft Visual C++ Redistributable
- Administrator privileges for some automation features

#### Linux
- X11 or Wayland display server
- Required system packages (varies by distribution)
- Appropriate permissions for input devices

## Installation Methods

### Method 1: Automatic Installation (Recommended)

#### Using the Installation Script
```bash
# Clone the repository
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1

# Run the automatic installation script
./scripts/install/detect_os_and_install.sh
```

The script will:
1. Detect your operating system
2. Install system dependencies
3. Create a Python virtual environment
4. Install Python packages
5. Configure permissions
6. Run initial setup

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
```

#### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Install Package
```bash
pip install -e .
```

#### Step 5: Platform-Specific Setup

##### macOS
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Grant accessibility permissions
# System Preferences > Security & Privacy > Accessibility > Add Terminal/Python
```

##### Windows
```bash
# Install Microsoft Visual C++ Redistributable (if not already installed)
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

##### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y scrot python3-tk python3-dev
sudo apt install -y xvfb x11-utils
```

##### Linux (CentOS/RHEL)
```bash
sudo yum update
sudo yum install -y python3-devel python3-pip
sudo yum install -y scrot tkinter python3-devel
sudo yum install -y xorg-x11-server-Xvfb
```

### Method 3: Docker Installation

#### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1

# Start with Docker Compose
docker-compose --profile production up -d
```

#### Using Docker Directly
```bash
# Pull the image
docker pull ai-agent:latest

# Run the container
docker run -d \
  --name ai-agent \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  ai-agent:latest
```

## Configuration

### Basic Configuration
1. Copy the example configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

2. Edit the configuration file:
```yaml
api:
  preferred_provider: "ollama"  # or "openai", "anthropic"
  local_endpoint: "http://localhost:11434"
  local_model: "gemini-3-flash-preview:latest"
  logging:
  level: "INFO"
  console: true
  file: "logs/ai_agent.log"
```

### Environment Variables
Create a `.env` file or set environment variables:

```bash
# API Keys (if using cloud providers)
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Configuration
export AI_AGENT_CONFIG_PATH="./config/config.yaml"
export AI_AGENT_LOG_LEVEL="INFO"

# Ollama Configuration (if using local models)
export OLLAMA_ENDPOINT="http://localhost:11434"
```

## Verification

### Test Installation
```bash
# Run health check
ai-agent-validate

# Test screenshot capture
python -m ai_agent.platform_abstraction.screenshot_capture --test

# Test GUI automation
python -m ai_agent.platform_abstraction.gui_automation --test
```

### Run Basic Test
```bash
# Simple test command
ai-agent "Take a screenshot and save it to test.png"
```

## Troubleshooting

### Common Issues

#### Permission Denied (macOS)
```bash
# Grant accessibility permissions
# System Preferences > Security & Privacy > Privacy > Accessibility
# Add Terminal and/or Python to the allowed applications
```

#### Display Issues (Linux)
```bash
# If using X11 forwarding or remote display
export DISPLAY=:0
# Or use Xvfb for headless operation
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
```

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Docker Issues
```bash
# Check if Docker is running
docker info

# Check X11 forwarding (Linux/macOS)
echo $DISPLAY
xhost +local:docker
```

### Getting Help

1. Check the logs: `tail -f logs/ai_agent.log`
2. Run diagnostics: `ai-agent-validate`
3. Check GitHub Issues
4. Create a new issue with:
   - Operating system and version
   - Python version
   - Error messages
   - Configuration file (sanitized)

## Next Steps

After successful installation:

1. Read the [Architecture Documentation](ARCHITECTURE.md)
2. Review the [Configuration Guide](CONFIGURATION.md)
3. Try the [Quick Start Examples](EXAMPLES.md)
4. Explore the [API Documentation](API.md)

## Uninstallation

### Manual Uninstallation
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove package
pip uninstall ai-agent

# Remove configuration and logs (optional)
rm -rf config/ logs/
```

### Docker Uninstallation
```bash
# Stop and remove container
docker stop ai-agent
docker rm ai-agent

# Remove image
docker rmi ai-agent:latest
```

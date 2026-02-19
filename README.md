<div align="center">

# VEXIS-1.1

**Vision-based AI Agent for GUI Automation**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()

*AI-powered visual understanding and intelligent automation*

</div>

VEXIS-1.1 is an advanced AI agent that combines computer vision with intelligent automation to interact with graphical user interfaces naturally. By understanding screen content and context, VEXIS-1.1 can perform complex tasks across applications with human-like precision.

## 🚀 Quick Start

Get VEXIS-1.1 running in seconds with zero configuration:

```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
python3 run.py "take a screenshot"
```

**Zero setup required** - VEXIS-1.1 automatically handles environment creation, dependency installation, and configuration.

### 💡 Capabilities

```bash
# Mathematical operations
python3 run.py "open calculator and compute 2+2"

# Web navigation and search
python3 run.py "open browser and search for AI automation"

# Document creation and editing
python3 run.py "create text document, write 'Hello World', and save"
```

---

## ⚠️ Safety Notice

**VEXIS-1.1 performs real operations on your system**  
VEXIS-1.1 can execute actions including file deletion, purchases, and email sending. **Use responsibly** and maintain supervision during operation.

## ✨ Core Features

### � Advanced Vision Intelligence
- **Visual Context Understanding**: Interprets screen content with semantic awareness
- **Adaptive Decision Making**: Selects optimal actions based on current interface state
- **Learning from Experience**: Improves execution strategies through interaction patterns

### ⚡ Two-Phase Execution Architecture
1. **Strategic Planning**: Decomposes complex instructions into actionable steps
2. **Precision Execution**: Implements tasks with real-time visual feedback and verification

### 🌐 Universal Platform Support
- **macOS**: Native integration with full feature support
- **Windows**: Comprehensive compatibility and testing
- **Linux**: Active development and community testing

## 📖 Documentation

→ **[USAGE.md](USAGE.md)** - Comprehensive usage guide  
→ **[Docker Guide](docker/README.md)** - Container deployment methods

---

## �️ Installation

### ⚡ Automated Setup (Recommended)

```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
python3 run.py "take a screenshot"
```

**Automatically configured:**
- ✅ Virtual environment creation and activation
- ✅ Dependency resolution and installation
- ✅ Configuration file initialization
- ✅ System validation and testing

### 🔧 Manual Installation

```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
pip install -e .
```

### 🤖 AI Model Integration

VEXIS-1.1 supports multiple AI providers for different use cases:

#### Local Processing (Ollama)
- Install [Ollama](https://ollama.ai/download) for local model execution
- Initialize with `ollama signin` for cloud model access
- Benefits: Complete privacy, offline operation, no ongoing costs

#### Cloud Processing (Google Gemini)
- Obtain API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Leverage Google's infrastructure for enhanced performance
- Benefits: Higher accuracy, scalability, managed infrastructure

---

## 🎮 Usage

### Basic Interface
```bash
# Natural language commands
python3 run.py "your instruction here"

# Enhanced debugging mode
python3 run.py "instruction" --debug
```

---

## 🧠 Model Architecture

**VEXIS-1.1 integrates with leading AI providers:**

### 🦙 Ollama (Local Intelligence)
- **Architecture**: Local model deployment with privacy-first design
- **Models**: Gemini 3 Flash Preview and other open-source models
- **Advantages**: Data sovereignty, offline capability, cost efficiency

### 🔵 Google Gemini (Cloud Intelligence)
- **Architecture**: Google's production-grade API infrastructure
- **Models**: Gemini 1.5 Flash with enterprise-grade reliability
- **Advantages**: Enhanced accuracy, automatic scaling, managed reliability

### 🎯 Provider Selection Strategy

**Development & Prototyping**: Ollama provides rapid iteration without costs  
**Production & Critical Workflows**: Google Gemini ensures reliability and performance

## ⚙️ Configuration

### Runtime Configuration

VEXIS-1.1 uses `config.yaml` for system-wide settings:

```yaml
api:
  preferred_provider: "ollama"  # "ollama" or "google"
  local_endpoint: "http://localhost:11434"
  local_model: "gemini-3-flash-preview:latest"
  timeout: 120
  max_retries: 3
  # API keys are managed interactively for security
```

### Command Interface
```bash
python3 run.py "instruction" --help           # Complete option reference
python3 run.py "instruction" --debug          # Verbose execution logging
python3 run.py "instruction" --no-prompt      # Skip provider selection
```

## 🏗️ System Architecture

**Two-Phase Execution Model:**
1. **Cognitive Planning**: Analyzes instructions and generates structured task sequences
2. **Adaptive Execution**: Implements tasks with continuous visual feedback and error recovery

**Core Components:**
- `TwoPhaseEngine` - Central orchestration and execution management
- `ScreenshotCapture` - High-fidelity screen acquisition and processing
- `GUIAutomation` - Precise input simulation and interface interaction
- `ModelRunner` - Unified AI provider abstraction layer

## 📦 Technical Requirements

**Automated Dependency Management**

VEXIS-1.1's setup system automatically configures all required components:

### Core Dependencies
- `Pillow>=10.0.0` - Advanced image processing and manipulation
- `pyautogui>=0.9.54` - Cross-platform GUI automation framework
- `opencv-python>=4.8.0` - Computer vision and image analysis
- `numpy>=1.24.0` - High-performance numerical computing
- `pynput>=1.7.6` - System-level input device control

### AI/ML Integration
- `openai>=1.0.0` - OpenAI API client library
- `anthropic>=0.7.0` - Anthropic Claude API integration
- `transformers>=4.35.0` - Hugging Face model ecosystem
- `torch>=2.1.0` - PyTorch deep learning framework

### System Integration
- `cryptography>=41.0.0` - Secure communications and data protection
- `pydantic>=2.0.0` - Data validation and type safety
- `structlog>=23.0.0` - Structured logging and observability
- `rich>=13.0.0` - Advanced terminal interface and formatting

### Platform-Specific Components
- `pyobjc-framework-Cocoa>=9.0` (macOS) - Native macOS integration
- `pywin32>=306` (Windows) - Windows API access
- `python-xlib>=0.33` (Linux) - X11 window system interface

## � Troubleshooting

### Common Resolution Patterns

#### Ollama Integration
- **Connection Issues**: Verify Ollama service at `http://localhost:11434`
- **Authentication**: Ensure valid account with `ollama signin`
- **Model Availability**: Confirm `gemini-3-flash-preview:latest` accessibility

#### Google API Integration
- **API Key Validation**: Generate valid credentials from Google AI Studio
- **Quota Management**: Monitor usage against free tier limitations
- **Network Connectivity**: Ensure stable internet connection

#### System-Level Issues
- **Permission Errors**: Configure accessibility permissions on macOS
- **Dependency Conflicts**: Reinstall using automated setup process
- **Configuration Reset**: Clear settings in `~/.vexis/settings.json`

### Support Resources
- [GitHub Issues](https://github.com/AInohogosya/VEXIS-1/issues) - Problem reporting and tracking
- System logs provide detailed diagnostic information
- Environment validation tools available in debug mode

## 🧪 Development Environment

```bash
pytest              # Execute test suite
black src/          # Code formatting and style enforcement
mypy src/           # Static type analysis and validation
```

## 📄 License & Contributions

- **License**: MIT License - see [LICENSE](LICENSE) for complete terms
- **Contribution Process**: Fork → Feature Branch → Testing → Pull Request
- **Code Standards**: Automated formatting, type checking, and test coverage

## 📚 Additional Resources

- **Inline Documentation**: Comprehensive code-level documentation
- **Configuration Reference**: Detailed `config.yaml` options
- **Command Reference**: Complete CLI interface documentation via `--help`

## 🛠️ Development Tools

```bash
# Post-installation commands
vexis-1              # Standard AI agent interface
vexis-1-enhanced     # Advanced two-phase execution engine
pytest              # Comprehensive test execution
black src/          # Automated code formatting
mypy src/           # Static type checking and analysis
```

---

<div align="center">

**⚡ VEXIS-1.1 represents a new paradigm in intelligent automation. Deploy with care and verify critical operations.**

**🌟 Support the project by giving us a star on GitHub!**

[GitHub Repository](https://github.com/AInohogosya/VEXIS-1) | [Issue Tracker](https://github.com/AInohogosya/VEXIS-1/issues) | [Community Discussions](https://github.com/AInohogosya/VEXIS-1/discussions)

</div>

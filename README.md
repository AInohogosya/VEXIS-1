<img width="1280" height="720" alt="VEXIS-1" src="https://github.com/user-attachments/assets/7612c621-6dfc-4f15-b968-a80772268032" />


VEXIS-1 is a powerful vision-based AI agent for GUI automation that uses computer vision to understand and interact with graphical user interfaces.

## 🏛️ Name Origin

The name "VEXIS" is derived from the Latin word *Vexillum* (military standard/guiding flag). It represents a "flag bearer" that raises the user's intentions and guides the system through chaotic OS environments.

## 🤖 Features

- **Vision-based automation**: Uses computer vision to understand GUI elements
- **Two-phase architecture**: Task list generation + sequential execution
- **Cross-platform support**: Works on macOS, Windows, and Linux
- **Multiple AI models**: Supports various AI models including Gemini 3 Flash
- **Screenshot capture**: Takes and analyzes screenshots for visual understanding
- **GUI automation**: Performs clicks, typing, scrolling, and other interactions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) account and installation
- Ollama running locally on your machine
- Gemini 3 Flash model pulled in Ollama

### Platform Support

**Note**: This project has been primarily tested on macOS. While it is designed to work on Windows and Linux as well, cross-platform compatibility may vary. The codebase includes platform-specific dependencies for all three major operating systems, but testing has been focused on macOS environments.

### Installation

#### Step 1: Set up Ollama

1. **Create an Ollama account**:
   - Visit [https://ollama.ai/](https://ollama.ai/)
   - Sign up for a free account

2. **Install Ollama**:
   - Download and install Ollama for your operating system from [https://ollama.ai/download](https://ollama.ai/download)
   - Follow the installation instructions for your platform

3. **Start Ollama service**:
   - Open your terminal/command prompt
   - Run: `ollama serve`
   - Verify it's running at `http://localhost:11434`

4. **Pull the required model**:
   ```bash
   ollama pull gemini-3-flash-preview:latest
   ```

#### Step 2: Clone and Install VEXIS-1

1. Clone this repository:
```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

**Note**: The program will automatically check for missing dependencies and prompt you to install them if needed.

### Usage

Run VEXIS-1 with a simple command:

```bash
python3 run.py "your instruction here"
```

**Examples:**
```bash
python3 run.py "Take a screenshot"
python3 run.py "Open the calculator and perform 2+2"
python3 run.py "Open a web browser and navigate to google.com"
```

For debug mode:
```bash
python3 run.py "your instruction" --debug
```

## 🧠 AI Models Used

This project primarily uses:
- **Gemini 3 Flash** (via Ollama) - Main model for vision-based tasks
- **Local inference** through Ollama at `http://localhost:11434`

## 📋 Dependencies

### Core Dependencies
- `Pillow>=10.0.0` - Image processing
- `pyautogui>=0.9.54` - GUI automation
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical computations
- `pynput>=1.7.6` - Input device control

### AI/ML Dependencies
- `openai>=1.0.0` - OpenAI API support
- `anthropic>=0.7.0` - Anthropic API support
- `transformers>=4.35.0` - Hugging Face transformers
- `torch>=2.1.0` - PyTorch

### Platform-specific Dependencies
- `pyobjc-framework-Cocoa>=9.0` (macOS)
- `pywin32>=306` (Windows)
- `python-xlib>=0.33` (Linux)

## ⚙️ Configuration

The agent can be configured via `config.yaml`:

```yaml
api:
  preferred_provider: "ollama"
  local_endpoint: "http://localhost:11434"
  local_model: "gemini-3-flash-preview:latest"
  timeout: 120
  max_retries: 3
```

## 🏗️ Architecture

The project uses a two-phase architecture:

1. **Task Generation Phase**: Analyzes the instruction and generates a task list
2. **Execution Phase**: Sequentially executes each task with visual feedback

### Key Components

- `TwoPhaseEngine` - Core execution engine
- `ScreenshotCapture` - Handles screenshot capture and processing
- `GUIAutomation` - Manages GUI interactions
- `ModelRunner` - Interfaces with AI models

## ⚠️ **IMPORTANT DISCLAIMER**

**WARNING**: VEXIS-1 has the capability to perform automated actions on your computer, including but not limited to:

- Deleting files and folders
- Installing or uninstalling software
- Making online purchases
- Sending emails or messages
- Modifying system settings
- Executing arbitrary commands

**The developers of this project are NOT responsible for any damage, data loss, financial loss, or any other consequences that may result from using VEXIS-1.**

**Use at your own risk**. Always:
- Monitor the agent's actions closely
- Test in a safe environment first
- Keep backups of important data
- Never run unattended with critical systems

## 🧪 Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black src/
```

### Type Checking
```bash
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

1. **Ollama connection failed**: Make sure Ollama is running at `http://localhost:11434`
2. **Model not found**: Ensure you've pulled the Gemini 3 Flash model using `ollama pull gemini-3-flash-preview:latest`
3. **Permission denied**: On macOS, you may need to grant accessibility permissions for screen recording and keyboard control
4. **Import errors**: Make sure all dependencies are installed correctly

### Getting Help

- Check the [Issues](https://github.com/YOUR_USERNAME/VEXIS-1/issues) page
- Review the logs for detailed error messages
- Ensure your environment meets all prerequisites

## 📚 Documentation

For more detailed documentation, please visit the `docs/` directory or check the inline documentation in the source code.

---

**Remember**: VEXIS-1 is a powerful automation tool. Use responsibly and always verify actions before running them in production environments.

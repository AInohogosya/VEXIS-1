# VEXIS-1

VEXIS-1 is a powerful vision-based AI agent for GUI automation that uses computer vision to understand and interact with graphical user interfaces.

## 🎯 Try It Now (No Setup Required)

```bash
python3 run.py "Take a screenshot"
```

That's it! VEXIS-1 automatically handles setup and uses cloud models. No installation or account required.

**More examples:**
```bash
python3 run.py "Open calculator and do 2+2"
python3 run.py "Open browser and go to google.com"
```

---

## 📋 What You Need to Know

### ⚠️ **Safety First**
VEXIS-1 can perform real actions on your computer - delete files, make purchases, send emails, etc. **Use at your own risk** and always monitor its actions.

### 🖥️ Platform Support
- **Primary**: macOS (fully tested)
- **Secondary**: Windows, Linux (designed to work, less tested)

### 🤖 How It Works
1. **Vision-based**: Uses computer vision to see and understand your screen
2. **Two-phase**: Plans tasks first, then executes them step-by-step
3. **AI-powered**: Uses smart models to figure out what to do

### 💻 What It Can Do
- Take screenshots and analyze them
- Open and use applications (calculator, browser, etc.)
- Click buttons, type text, scroll
- Navigate websites and interfaces
- Automate repetitive GUI tasks

---

## 🚀 Installation

### Quick Install (Recommended)

```bash
git clone https://github.com/AInohogosya/VEXIS-1.git
cd VEXIS-1
pip install -r requirements.txt
pip install -e .
```

That's it! You're ready to go.

### Local Models Setup (Optional)

If you want to use local AI models instead of cloud:

1. Install [Ollama](https://ollama.ai/download)
2. Start Ollama service
3. Run `ollama signin` for cloud model access (optional)

---

## 💻 Usage

### Basic Commands
```bash
# Simple instruction
python3 run.py "your instruction here"

# Debug mode (see what it's thinking)
python3 run.py "your instruction" --debug
```

---

## 🧠 AI Models Used

This project primarily uses:
- **AI Models via Ollama** - Local and cloud-based models for vision-based tasks
- **Ollama API** for model inference

## ⚙️ Configuration

Configure via `config.yaml`:

```yaml
api:
  preferred_provider: "ollama"
  local_endpoint: "http://localhost:11434"
  local_model: "gemini-3-flash-preview:latest"
  timeout: 120
  max_retries: 3
```

## 🏗️ Architecture

1. **Task Generation**: Analyzes instruction and creates task list
2. **Execution**: Performs tasks step-by-step with visual feedback

**Key Components:**
- `TwoPhaseEngine` - Core execution engine
- `ScreenshotCapture` - Screen capture and processing
- `GUIAutomation` - Mouse/keyboard interactions
- `ModelRunner` - AI model interface

## 📋 Dependencies

**Core:**
- `Pillow>=10.0.0` - Image processing
- `pyautogui>=0.9.54` - GUI automation
- `opencv-python>=4.8.0` - Computer vision
- `numpy>=1.24.0` - Numerical computations
- `pynput>=1.7.6` - Input device control
- `ollama>=0.1.0` - Ollama client

**AI/ML:**
- `openai>=1.0.0` - OpenAI API
- `anthropic>=0.7.0` - Anthropic API
- `transformers>=4.35.0` - Hugging Face
- `torch>=2.1.0` - PyTorch

**Platform-specific:**
- `pyobjc-framework-Cocoa>=9.0` (macOS)
- `pywin32>=306` (Windows)
- `python-xlib>=0.33` (Linux)

## 🐛 Troubleshooting

**Common Issues:**
1. **Ollama connection**: Make sure Ollama runs at `http://localhost:11434`
2. **Cloud access denied**: Run `ollama signin` with valid account
3. **Model not found**: Check if `gemini-3-flash-preview:latest` is available in your region
4. **Permission denied**: Grant accessibility permissions on macOS
5. **Import errors**: Install missing dependencies

**Getting Help:**
- [Issues page](https://github.com/AInohogosya/VEXIS-1/issues)
- Check logs for detailed errors
- Verify environment setup

## 🧪 Development

```bash
pytest          # Run tests
black src/       # Format code
mypy src/        # Type checking
```

## 📄 License & Contributing

- **License**: MIT - see [LICENSE](LICENSE) file
- **Contributing**: Fork → Create branch → Make changes → Add tests → Pull request

## 📚 Documentation

Detailed docs in `docs/` directory and inline in source code (e.g., `src/ai_agent/core_processing/two_phase_engine.py`).

---

**Remember**: VEXIS-1 is a powerful automation tool. Use responsibly and always verify actions before running in production environments.

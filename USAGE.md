# Usage Guide

## Basic Execution

```bash
python3 run.py "your instruction here"
```

## Examples

### Simple Tasks
```bash
# Take a screenshot
python3 run.py "Take a screenshot"

# Open calculator and perform calculation
python3 run.py "Open the calculator and perform 2+2"

# Browse to a website
python3 run.py "Open a web browser and navigate to google.com"
```

### Advanced Tasks
```bash
# Multi-step automation
python3 run.py "Open a text editor, type 'Hello World', and save the file"

# Application interaction
python3 run.py "Open System Preferences and change the wallpaper"

# Web automation
python3 run.py "Open browser, search for 'AI automation', and take a screenshot"
```

## Debug Mode

```bash
python3 run.py "your instruction" --debug
```

Debug mode provides:
- Detailed logging of AI decisions
- Screenshot previews
- Step-by-step execution tracking
- Error analysis

## Configuration Options

The system can be configured via `config.yaml`:

```yaml
api:
  preferred_provider: "ollama"  # or "openai", "anthropic"
  local_model: "gemini-3-flash-preview:latest"
  timeout: 120

engine:
  click_delay: 0.1
  typing_delay: 0.05
  max_task_retries: 3
```

## Environment Variables

```bash
# Override API endpoint
export AI_AGENT_LOCAL_ENDPOINT="http://localhost:11434"

# Set log level
export AI_AGENT_LOG_LEVEL="DEBUG"

# Skip dependency checks
export AI_AGENT_SKIP_DEPS="true"
```

## Installation Validation

```bash
# Test basic installation
python3 run.py "Take a test screenshot" --debug

# Test installed commands
vexis-1 --help
vexis-1-enhanced --help

# Check system status
python3 -c "from ai_agent.utils.config import load_config; print('Config loaded successfully')"
```

## Getting Help

```bash
# Show all options for run.py
python3 run.py --help

# Show help for installed commands
vexis-1 --help
vexis-1-enhanced --help

# Check configuration
python3 -c "from ai_agent.utils.config import load_config; print(load_config())"
```

## Tips

1. **Start simple**: Begin with basic tasks to understand the system
2. **Use debug mode**: For complex tasks, use `--debug` to see what's happening
3. **Be specific**: Clear, specific instructions work better
4. **Monitor closely**: Always watch the automation, especially for critical tasks
5. **Check logs**: Review logs for troubleshooting and understanding

"""
Yellow Selection System Configuration
Reproducible settings for consistent behavior across platforms
"""

# Color Configuration - Reproducible Yellow Highlighting
YELLOW_SELECTION_CONFIG = {
    # Colors (ANSI codes for consistency)
    'colors': {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'BLACK': '\033[30m',
        'YELLOW': '\033[33m',
        'BRIGHT_YELLOW': '\033[93m',
        'BG_YELLOW': '\033[43m',
        'WHITE': '\033[97m',
        'BRIGHT_WHITE': '\033[37m',
        'BRIGHT_GREEN': '\033[92m',
        'RED': '\033[91m',
        'CYAN': '\033[36m',
        'BRIGHT_CYAN': '\033[96m'
    },
    
    # Navigation Configuration
    'navigation': {
        'arrow_keys': {
            'up': ['\033[A', '\033OA', 'H', 'k', 'K'],
            'down': ['\033[B', '\033OB', 'P', 'j', 'J'],
            'enter': ['\r', '\n'],
            'quit': ['q', 'Q']
        },
        'number_keys': {
            'min': 1,
            'max': 9
        }
    },
    
    # Display Configuration
    'display': {
        'clear_screen_command': '\033[2J\033[H',
        'cursor_save': '\033[s',
        'cursor_restore': '\033[u',
        'line_clear': '\033[K',
        'move_cursor': '\033[{line};0H'
    },
    
    # Model Selection Configuration
    'model_selection': {
        'families': {
            'microsoft': {
                'name': 'Microsoft',
                'icon': '🔷',
                'description': "Microsoft's Phi family models"
            },
            'google': {
                'name': 'Google', 
                'icon': '🔍',
                'description': "Google's Gemma family models"
            },
            'alibaba': {
                'name': 'Alibaba',
                'icon': '🐲', 
                'description': "Alibaba's Qwen family models"
            },
            'meta': {
                'name': 'Meta',
                'icon': '🦙',
                'description': "Meta's Llama family models"
            }
        }
    },
    
    # Reproducibility Settings
    'reproducibility': {
        'consistent_colors': True,
        'cross_platform': True,
        'fallback_enabled': True,
        'debug_mode': False
    }
}


def get_config():
    """Get the yellow selection configuration"""
    return YELLOW_SELECTION_CONFIG


def get_colors():
    """Get color definitions"""
    return YELLOW_SELECTION_CONFIG['colors']


def get_navigation_config():
    """Get navigation configuration"""
    return YELLOW_SELECTION_CONFIG['navigation']


def get_display_config():
    """Get display configuration"""
    return YELLOW_SELECTION_CONFIG['display']


def is_reproducible_mode():
    """Check if reproducible mode is enabled"""
    return YELLOW_SELECTION_CONFIG['reproducibility']['consistent_colors']

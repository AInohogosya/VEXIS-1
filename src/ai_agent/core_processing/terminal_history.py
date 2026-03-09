#!/usr/bin/env python3
"""
Terminal History System for VEXIS-1.2
Replaces Save command with terminal log display and history preservation
OS-independent implementation optimized for VEXIS-1.2 architecture
"""

import json
import time
import os
import subprocess
import shlex
import platform
import stat
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path, PurePath
from enum import Enum
from contextlib import contextmanager

from ..utils.logger import get_logger
from ..utils.exceptions import ExecutionError, PlatformError, ValidationError


class TerminalEntryType(Enum):
    """Types of terminal entries"""
    COMMAND = "command"
    OUTPUT = "output"
    ERROR = "error"


@dataclass
class TerminalEntry:
    """Individual terminal entry with command execution information"""
    timestamp: float
    entry_type: TerminalEntryType
    content: str
    command: Optional[str] = None
    working_directory: Optional[str] = None
    return_code: Optional[int] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        data = asdict(self)
        data['entry_type'] = self.entry_type.value
        data['timestamp'] = str(self.timestamp)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TerminalEntry':
        """Create from dictionary"""
        data['entry_type'] = TerminalEntryType(data['entry_type'])
        data['timestamp'] = float(data['timestamp'])
        return cls(**data)


class TerminalHistory:
    """Manages terminal command history and execution logs"""
    
    def __init__(self, history_file: Optional[str] = None, max_entries: int = 1000):
        self.logger = get_logger("terminal_history")
        self.max_entries = max_entries
        self.entries: List[TerminalEntry] = []
        
        # Set history file path
        if history_file:
            self.history_file = Path(history_file)
        else:
            # Default to user's home directory
            self.history_file = Path.home() / ".vexis" / "terminal_history.json"
        
        self._ensure_history_dir()
        self._load_history()
    
    def _ensure_history_dir(self):
        """Ensure history directory exists"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            # Set appropriate permissions
            self.history_file.parent.chmod(0o700)
        except Exception as e:
            self.logger.warning(f"Could not create history directory: {e}")
    
    def _load_history(self):
        """Load history from file"""
        if not self.history_file.exists():
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.entries = []
            for entry_data in data.get('entries', []):
                try:
                    entry = TerminalEntry.from_dict(entry_data)
                    self.entries.append(entry)
                except Exception as e:
                    self.logger.warning(f"Could not load history entry: {e}")
            
            # Trim to max entries
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries:]
            
            self.logger.info(f"Loaded {len(self.entries)} history entries")
            
        except Exception as e:
            self.logger.error(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save history to file"""
        try:
            # Create backup before saving
            if self.history_file.exists():
                backup_file = self.history_file.with_suffix('.json.bak')
                self.history_file.rename(backup_file)
            
            # Prepare data for saving
            data = {
                'version': '1.0',
                'created_at': time.time(),
                'entries': [entry.to_dict() for entry in self.entries]
            }
            
            # Write to temporary file first
            temp_file = self.history_file.with_suffix('.json.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Move temporary file to final location
            temp_file.replace(self.history_file)
            
            # Set appropriate permissions
            self.history_file.chmod(0o600)
            
            self.logger.debug(f"Saved {len(self.entries)} history entries")
            
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
    
    def add_command(self, command: str, working_directory: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a command entry"""
        entry = TerminalEntry(
            timestamp=time.time(),
            entry_type=TerminalEntryType.COMMAND,
            content=command,
            command=command,
            working_directory=working_directory or os.getcwd(),
            metadata=metadata or {}
        )
        
        self.entries.append(entry)
        self._trim_if_needed()
        self._save_history()
        
        return len(self.entries) - 1
    
    def add_output(self, content: str, command_index: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add an output entry"""
        entry = TerminalEntry(
            timestamp=time.time(),
            entry_type=TerminalEntryType.OUTPUT,
            content=content,
            metadata=metadata or {}
        )
        
        if command_index is not None and 0 <= command_index < len(self.entries):
            # Link to command
            entry.metadata['command_index'] = command_index
            entry.command = self.entries[command_index].command
            entry.working_directory = self.entries[command_index].working_directory
        
        self.entries.append(entry)
        self._trim_if_needed()
        self._save_history()
        
        return len(self.entries) - 1
    
    def add_error(self, content: str, command_index: Optional[int] = None,
                  return_code: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add an error entry"""
        entry = TerminalEntry(
            timestamp=time.time(),
            entry_type=TerminalEntryType.ERROR,
            content=content,
            return_code=return_code,
            metadata=metadata or {}
        )
        
        if command_index is not None and 0 <= command_index < len(self.entries):
            # Link to command
            entry.metadata['command_index'] = command_index
            entry.command = self.entries[command_index].command
            entry.working_directory = self.entries[command_index].working_directory
        
        self.entries.append(entry)
        self._trim_if_needed()
        self._save_history()
        
        return len(self.entries) - 1
    
    def _trim_if_needed(self):
        """Trim entries if exceeding max"""
        if len(self.entries) > self.max_entries:
            # Remove oldest entries
            excess = len(self.entries) - self.max_entries
            self.entries = self.entries[excess:]
    
    def get_recent_commands(self, count: int = 10) -> List[TerminalEntry]:
        """Get recent command entries"""
        commands = [entry for entry in self.entries 
                   if entry.entry_type == TerminalEntryType.COMMAND]
        return commands[-count:] if len(commands) > count else commands
    
    def get_command_history(self, command_filter: Optional[str] = None) -> List[str]:
        """Get command history as list of strings"""
        commands = []
        for entry in self.entries:
            if entry.entry_type == TerminalEntryType.COMMAND:
                if command_filter is None or command_filter in entry.command:
                    commands.append(entry.command)
        return commands
    
    def get_session_entries(self, session_id: Optional[str] = None) -> List[TerminalEntry]:
        """Get entries for a specific session"""
        if session_id is None:
            # Get entries from last hour
            cutoff_time = time.time() - 3600
            return [entry for entry in self.entries if entry.timestamp >= cutoff_time]
        
        return [entry for entry in self.entries 
                if entry.metadata.get('session_id') == session_id]
    
    def display_history(self, count: int = 20, show_output: bool = False):
        """Display terminal history in a formatted way"""
        recent_entries = self.entries[-count:] if len(self.entries) > count else self.entries
        
        print("\n" + "="*80)
        print("📋 VEXIS Terminal History")
        print("="*80)
        
        current_command = None
        
        for i, entry in enumerate(recent_entries):
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(entry.timestamp))
            
            if entry.entry_type == TerminalEntryType.COMMAND:
                current_command = i
                print(f"\n{timestamp_str} 💻 {entry.working_directory or os.getcwd()}")
                print(f"$ {entry.content}")
                
            elif entry.entry_type == TerminalEntryType.OUTPUT:
                if show_output and current_command is not None:
                    print(f"{timestamp_str} 📤 Output:")
                    for line in entry.content.split('\n')[:10]:  # Limit output lines
                        print(f"   {line}")
                    if len(entry.content.split('\n')) > 10:
                        print("   ... (truncated)")
                        
            elif entry.entry_type == TerminalEntryType.ERROR:
                print(f"{timestamp_str} ❌ Error:")
                if entry.return_code is not None:
                    print(f"   Exit code: {entry.return_code}")
                for line in entry.content.split('\n')[:5]:  # Limit error lines
                    print(f"   {line}")
                if len(entry.content.split('\n')) > 5:
                    print("   ... (truncated)")
        
        print("\n" + "="*80)
    
    def search_history(self, query: str, entry_type: Optional[TerminalEntryType] = None) -> List[TerminalEntry]:
        """Search history for specific content"""
        results = []
        
        for entry in self.entries:
            if entry_type and entry.entry_type != entry_type:
                continue
            
            if query.lower() in entry.content.lower():
                results.append(entry)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics"""
        commands = [entry for entry in self.entries if entry.entry_type == TerminalEntryType.COMMAND]
        outputs = [entry for entry in self.entries if entry.entry_type == TerminalEntryType.OUTPUT]
        errors = [entry for entry in self.entries if entry.entry_type == TerminalEntryType.ERROR]
        
        # Most common commands
        command_counts = {}
        for command in commands:
            cmd_name = command.command.split()[0] if command.command else ""
            command_counts[cmd_name] = command_counts.get(cmd_name, 0) + 1
        
        most_common = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_entries': len(self.entries),
            'commands': len(commands),
            'outputs': len(outputs),
            'errors': len(errors),
            'most_common_commands': most_common,
            'date_range': {
                'oldest': min((e.timestamp for e in self.entries), default=None),
                'newest': max((e.timestamp for e in self.entries), default=None)
            }
        }
    
    def clear_history(self, confirm: bool = False):
        """Clear terminal history"""
        if not confirm:
            response = input("Are you sure you want to clear terminal history? (y/N): ")
            if response.lower() != 'y':
                return
        
        self.entries = []
        try:
            if self.history_file.exists():
                self.history_file.unlink()
            self.logger.info("Terminal history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")
    
    def export_history(self, export_path: str, format_type: str = "json"):
        """Export history to file"""
        try:
            export_file = Path(export_path)
            
            if format_type.lower() == "json":
                data = {
                    'exported_at': time.time(),
                    'entries': [entry.to_dict() for entry in self.entries]
                }
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format_type.lower() == "txt":
                with open(export_file, 'w', encoding='utf-8') as f:
                    for entry in self.entries:
                        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.timestamp))
                        f.write(f"[{timestamp_str}] {entry.entry_type.value.upper()}: {entry.content}\n")
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"History exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting history: {e}")
            raise


@contextmanager
def command_recorder(history: TerminalHistory, command: str, working_dir: Optional[str] = None):
    """Context manager for recording command execution"""
    command_index = history.add_command(command, working_dir)
    start_time = time.time()
    
    try:
        yield command_index
        
        # Record successful completion
        duration = time.time() - start_time
        history.add_output(
            f"Command completed successfully in {duration:.2f} seconds",
            command_index,
            metadata={'duration': duration}
        )
        
    except subprocess.CalledProcessError as e:
        # Record command error
        duration = time.time() - start_time
        history.add_error(
            str(e),
            command_index,
            return_code=e.returncode,
            metadata={'duration': duration}
        )
        raise
    
    except Exception as e:
        # Record general error
        duration = time.time() - start_time
        history.add_error(
            str(e),
            command_index,
            metadata={'duration': duration}
        )
        raise


if __name__ == "__main__":
    # Test terminal history
    history = TerminalHistory()
    
    # Add some test entries
    history.add_command("ls -la", "/home/user")
    history.add_output("file1.txt\nfile2.py\ndirectory/", metadata={'lines': 3})
    history.add_command("python test.py", "/home/user")
    history.add_error("FileNotFoundError: No such file", return_code=1)
    
    # Display history
    history.display_history()
    
    # Show statistics
    stats = history.get_statistics()
    print(f"\nStatistics: {stats}")

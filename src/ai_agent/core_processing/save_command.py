"""
Save Command Implementation for Phase 2: Execution Engine
Implements the save() command system for work logging and reflection
"""

import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from ..utils.logger import get_logger
from ..utils.exceptions import ExecutionError


class SaveContentType(Enum):
    """Types of save content"""
    FEEDBACK = "feedback"
    EXTRACTION = "extraction" 
    FAILURE = "failure"


@dataclass
class SaveEntry:
    """Individual save entry with work log information"""
    timestamp: float
    content: str
    content_type: SaveContentType
    operation_command: Optional[str] = None
    coordinates: Optional[tuple] = None
    visual_feedback: Optional[str] = None
    extracted_info: Optional[Dict[str, Any]] = None
    failure_details: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkLog:
    """Complete work log for a session"""
    session_id: str
    start_time: float
    entries: List[SaveEntry] = field(default_factory=list)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SaveCommand:
    """
    save command Implementation for Phase 2: Execution Engine
    Implements the save() command system for work logging and reflection
    
    The save command must be recognized simply as "save" - not "Save Command" or similar.
    """
    
    def __init__(self, session_id: Optional[str] = None, log_dir: str = "./work_logs"):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger("save_command")
        
        # Initialize work log
        self.work_log = WorkLog(
            session_id=self.session_id,
            start_time=time.time()
        )
        
        # Previous save content for reflection
        self._previous_save_content: Optional[str] = None
        self._previous_save_entry: Optional[SaveEntry] = None
        
        self.logger.info(f"Save command system initialized for session: {self.session_id}")
    
    def save(self, content: str, **kwargs) -> bool:
        """
        Main save command implementation
        
        Args:
            content: The content to save (work log information)
            **kwargs: Additional parameters including:
                - operation_command: The command that was executed
                - coordinates: Coordinates used in the operation
                - visual_feedback: Visual feedback from the operation
                - extracted_info: Dictionary of extracted information
                - failure_details: Dictionary of failure information
                - content_type: Type of content (feedback, extraction, failure)
        
        Returns:
            bool: True if save successful
        """
        try:
            # Determine content type
            content_type = kwargs.get('content_type', SaveContentType.FEEDBACK)
            if isinstance(content_type, str):
                content_type = SaveContentType(content_type)
            
            # Create save entry
            entry = SaveEntry(
                timestamp=time.time(),
                content=content,
                content_type=content_type,
                operation_command=kwargs.get('operation_command'),
                coordinates=kwargs.get('coordinates'),
                visual_feedback=kwargs.get('visual_feedback'),
                extracted_info=kwargs.get('extracted_info'),
                failure_details=kwargs.get('failure_details'),
                metadata=kwargs.get('metadata', {})
            )
            
            # Add to work log
            self.work_log.entries.append(entry)
            
            # Update previous save for reflection
            self._previous_save_content = content
            self._previous_save_entry = entry
            
            # Persist to disk
            self._persist_work_log()
            
            self.logger.debug(f"Save command executed: {content[:100]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Save command failed: {e}")
            return False
    
    def get_previous_save_content(self) -> Optional[str]:
        """Get content of immediately preceding save for reflection"""
        return self._previous_save_content
    
    def get_previous_save_entry(self) -> Optional[SaveEntry]:
        """Get complete entry of immediately preceding save"""
        return self._previous_save_entry
    
    def get_recent_saves(self, count: int = 5) -> List[SaveEntry]:
        """Get recent save entries for reflection"""
        return self.work_log.entries[-count:] if self.work_log.entries else []
    
    def has_failures(self) -> bool:
        """Check if there are any failure entries in the work log"""
        return any(entry.content_type == SaveContentType.FAILURE for entry in self.work_log.entries)
    
    def get_failure_coordinates(self) -> List[tuple]:
        """Get list of coordinates that failed to prevent repeated clicking"""
        failure_coords = []
        for entry in self.work_log.entries:
            if (entry.content_type == SaveContentType.FAILURE and 
                entry.coordinates and 
                entry.failure_details):
                failure_coords.append(entry.coordinates)
        return failure_coords
    
    def get_extracted_information(self) -> Dict[str, Any]:
        """Get all extracted information from work log"""
        extracted = {}
        for entry in self.work_log.entries:
            if entry.extracted_info:
                extracted.update(entry.extracted_info)
        return extracted
    
    
    def _persist_work_log(self) -> None:
        """Persist work log to disk"""
        try:
            log_file = self.log_dir / f"{self.session_id}.json"
            
            # Convert to serializable format
            log_data = {
                'session_id': self.work_log.session_id,
                'start_time': str(self.work_log.start_time),  # Convert to string
                'end_time': str(self.work_log.end_time) if self.work_log.end_time else None,  # Convert to string
                'metadata': self.work_log.metadata,
                'entries': []
            }
            
            for entry in self.work_log.entries:
                entry_data = asdict(entry)
                # Convert enum to string and timestamp to string
                entry_data['content_type'] = entry.content_type.value
                entry_data['timestamp'] = str(entry.timestamp)  # Convert to string
                log_data['entries'].append(entry_data)
            
            # Write to file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to persist work log: {e}")
    
    def end_session(self) -> None:
        """End the current session and finalize work log"""
        self.work_log.end_time = time.time()
        self._persist_work_log()
        self.logger.info(f"Session ended: {self.session_id}")
    
    def load_session(self, session_id: str) -> bool:
        """Load a previous session for reflection"""
        try:
            log_file = self.log_dir / f"{session_id}.json"
            if not log_file.exists():
                return False
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # Reconstruct work log
            self.work_log = WorkLog(
                session_id=log_data['session_id'],
                start_time=log_data['start_time'],
                end_time=log_data.get('end_time'),
                metadata=log_data.get('metadata', {})
            )
            
            # Reconstruct entries
            for entry_data in log_data.get('entries', []):
                entry_data['content_type'] = SaveContentType(entry_data['content_type'])
                entry = SaveEntry(**entry_data)
                self.work_log.entries.append(entry)
            
            # Update previous save for reflection
            if self.work_log.entries:
                self._previous_save_entry = self.work_log.entries[-1]
                self._previous_save_content = self._previous_save_entry.content
            
            self.session_id = session_id
            self.logger.info(f"Session loaded: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return False


# Global save command instance for easy access
_global_save_command: Optional[SaveCommand] = None


def get_save_command() -> SaveCommand:
    """Get global save command instance"""
    global _global_save_command
    if _global_save_command is None:
        _global_save_command = SaveCommand()
    return _global_save_command


def save(content: str, **kwargs) -> bool:
    """
    Global save function for easy use
    
    Example usage:
    save("Clicked the browser's search bar. Response received.", 
         operation_command="click(0.5, 0.2)",
         visual_feedback="Search bar highlighted",
         content_type="feedback")
    """
    return get_save_command().save(content, **kwargs)

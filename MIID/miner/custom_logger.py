from typing import Optional, List
import logging
import os

class CustomLogger:
    """Customized logging system with buffered output and name prefix formatting"""
    
    def __init__(self, name: str, output_file: Optional[str] = None, use_stdout: bool = True, debug_level: int = 0):
        self.name = name
        self.output_file = output_file
        self.use_stdout = use_stdout
        self.log_buffer = []
        self.tab_level = 0
        self.debug_level = debug_level
        
    def _format_message(self, level: str, message: str) -> str:
        """Format log message with name prefix and tab levels"""
        prefix = f"{self.name:20s}: "
        tabs = "  " * self.tab_level
        return f"{prefix}{tabs}{level}: {message}"
    
    def _add_to_buffer(self, level: str, message: str):
        """Add formatted message to buffer"""
        if self.debug_level <= logging._nameToLevel[level]:
            formatted_message = self._format_message(level, message)
            self.log_buffer.append(formatted_message)
            if self.use_stdout:
                print(formatted_message, flush=True)

    def info(self, message: str):
        """Log info message"""
        self._add_to_buffer("INFO", message)
    
    def debug(self, message: str):
        """Log debug message"""
        self._add_to_buffer("DEBUG", message)
    
    def warning(self, message: str):
        """Log warning message"""
        self._add_to_buffer("WARNING", message)
    
    def error(self, message: str):
        """Log error message"""
        self._add_to_buffer("ERROR", message)
    
    def trace(self, message: str):
        """Log trace message"""
        self._add_to_buffer("TRACE", message)
    
    def increase_tab(self):
        """Increase tab level"""
        self.tab_level += 1
    
    def decrease_tab(self):
        """Decrease tab level"""
        self.tab_level = max(0, self.tab_level - 1)
    
    def set_tab_level(self, level: int):
        """Set tab level"""
        self.tab_level = max(0, level)
    
    def flush(self):
        """Flush all buffered logs to output"""
        if not self.log_buffer:
            return
            
        output_lines = []
        for message in self.log_buffer:
            output_lines.append(message)
        
        # Output to stdout if enabled
        if self.use_stdout:
            for line in output_lines:
                print(line, flush=True)
        
        # Output to file if specified
        if self.output_file:
            try:
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                with open(self.output_file, "w", encoding="utf-8") as f:
                    for line in output_lines:
                        f.write(line + "\n")
            except Exception as e:
                print(f"Error writing to log file {self.output_file}: {e}", flush=True)
        
        # Clear buffer
        self.log_buffer.clear()
    
    def get_logs(self) -> List[str]:
        """Get all buffered logs without flushing"""
        return self.log_buffer.copy()
    
    def clear_buffer(self):
        """Clear the log buffer"""
        self.log_buffer.clear()

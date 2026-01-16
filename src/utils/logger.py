"""
Logging utility module for USB Camera Viewer
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Global logger registry
_loggers = {}


def setup_logger(
    name: str = "USBCameraViewer",
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to a file
        log_dir: Directory for log files. If None, uses ./logs
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{name.lower()}_{timestamp}.log"
        
        file_handler = logging.FileHandler(
            log_file, 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Store in registry
    _loggers[name] = logger
    
    logger.debug(f"Logger '{name}' initialized")
    
    return logger


def get_logger(name: str = "USBCameraViewer") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


def set_log_level(name: str, level: int) -> None:
    """
    Change the log level of an existing logger.
    
    Args:
        name: Logger name
        level: New logging level
    """
    if name in _loggers:
        logger = _loggers[name]
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)


class LoggerMixin:
    """
    Mixin class that provides logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info("MyClass initialized")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger named after the class."""
        name = self.__class__.__name__
        if name not in _loggers:
            setup_logger(name, log_to_file=False)
        return _loggers[name]


# Convenience functions for quick logging
def debug(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log a debug message."""
    get_logger(logger_name).debug(msg)


def info(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log an info message."""
    get_logger(logger_name).info(msg)


def warning(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log a warning message."""
    get_logger(logger_name).warning(msg)


def error(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log an error message."""
    get_logger(logger_name).error(msg)


def critical(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log a critical message."""
    get_logger(logger_name).critical(msg)


def exception(msg: str, logger_name: str = "USBCameraViewer") -> None:
    """Log an exception with traceback."""
    get_logger(logger_name).exception(msg)

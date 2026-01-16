"""
Configuration management module for USB Camera Viewer
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """
    Manages application configuration settings.
    Loads and saves settings to a JSON configuration file.
    """
    
    DEFAULT_CONFIG = {
        "default_resolution": [1280, 720],
        "default_fps": 30,
        "save_path": "./captures",
        "auto_connect": True,
        "window_size": [1024, 768],
        "fullscreen_on_start": False,
        "show_fps": True,
        "show_resolution": True,
        "video_format": "mp4",
        "image_format": "png",
        "brightness": 50,
        "contrast": 50,
        "saturation": 50,
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "last_device_index": 0,
        "theme": "dark"
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default config path in user's home directory or app directory
            app_dir = Path(__file__).parent.parent.parent
            config_path = app_dir / "config.json"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file or create with defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self._config = {**self.DEFAULT_CONFIG, **loaded_config}
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file: {e}")
                self._config = self.DEFAULT_CONFIG.copy()
        else:
            self._config = self.DEFAULT_CONFIG.copy()
            self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Could not save config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any, save: bool = True) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            save: Whether to save to file immediately
        """
        self._config[key] = value
        if save:
            self._save_config()
    
    def update(self, settings: Dict[str, Any], save: bool = True) -> None:
        """
        Update multiple configuration values.
        
        Args:
            settings: Dictionary of settings to update
            save: Whether to save to file immediately
        """
        self._config.update(settings)
        if save:
            self._save_config()
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to default values."""
        self._config = self.DEFAULT_CONFIG.copy()
        self._save_config()
    
    @property
    def default_resolution(self) -> tuple:
        """Get default resolution as tuple (width, height)."""
        res = self.get("default_resolution", [1280, 720])
        return tuple(res)
    
    @property
    def default_fps(self) -> int:
        """Get default FPS."""
        return self.get("default_fps", 30)
    
    @property
    def save_path(self) -> Path:
        """Get save path for captures."""
        path = Path(self.get("save_path", "./captures"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def window_size(self) -> tuple:
        """Get window size as tuple (width, height)."""
        size = self.get("window_size", [1024, 768])
        return tuple(size)
    
    @property
    def auto_connect(self) -> bool:
        """Get auto-connect setting."""
        return self.get("auto_connect", True)
    
    def __repr__(self) -> str:
        return f"Config({self.config_path})"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def init_config(config_path: Optional[str] = None) -> Config:
    """
    Initialize the global configuration with a specific path.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config

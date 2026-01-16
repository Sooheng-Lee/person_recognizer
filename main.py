#!/usr/bin/env python3
"""
USB Camera Viewer - Main Entry Point

A desktop application for viewing USB-connected cameras and smartphones
on your computer screen in real-time.

Usage:
    python main.py
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from src.ui.main_window import MainWindow
from src.utils.logger import setup_logger
from src.utils.config import init_config


def main():
    """Main entry point for the application."""
    
    # Initialize logging
    logger = setup_logger("USBCameraViewer")
    logger.info("=" * 50)
    logger.info("USB Camera Viewer starting...")
    logger.info("=" * 50)
    
    # Initialize configuration
    config = init_config()
    logger.info(f"Configuration loaded from: {config.config_path}")
    
    # Enable High DPI scaling for better display on modern monitors
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("USB Camera Viewer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("USBCameraViewer")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    logger.info("Application window displayed")
    
    # Run event loop
    exit_code = app.exec_()
    
    logger.info(f"Application exited with code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

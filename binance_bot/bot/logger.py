#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logger: centralise logs console + fichier."""
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "binance_bot", log_file: str = "logs/bot.log") -> logging.Logger:
    """
    Configure logger (console + fichier).
    
    Args:
        name: nom du logger
        log_file: chemin fichier log
    
    Returns:
        logger configuré
    """
    # Créer dossier logs si nécessaire
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Éviter doublons handlers
    if logger.handlers:
        return logger
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    # Test
    logger = setup_logger(log_file="../logs/test.log")
    logger.info("Test message INFO")
    logger.warning("Test message WARNING")
    logger.error("Test message ERROR")
    print("✅ Logger configuré. Vérifier logs/test.log")


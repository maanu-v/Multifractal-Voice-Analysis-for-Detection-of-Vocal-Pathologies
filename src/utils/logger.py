import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been added (prevents duplicate logs)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Prevent propagation to root logger to avoid double logging if root is config'd
        logger.propagate = False
        
    return logger

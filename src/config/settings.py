"""Configuration loading"""

import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()

    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

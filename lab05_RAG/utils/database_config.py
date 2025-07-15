"""
Database configuration utilities for Lab05 RAG system
Provides centralized PostgreSQL connection configuration
"""

import os
from typing import Dict, Any


def get_database_config() -> Dict[str, Any]:
    """
    Get PostgreSQL database configuration from environment variables
    
    Returns:
        Dict[str, Any]: Database configuration dictionary
    """
    return {
        'host': os.getenv('PG_HOST', 'localhost'),
        'port': os.getenv('PG_PORT', '5432'),
        'database': os.getenv('PG_DATABASE', 'labor_law_rag'),
        'user': os.getenv('PG_USER', 'postgres'),
        'password': os.getenv('PG_PASSWORD', 'your_password')
    }


def get_database_url() -> str:
    """
    Get PostgreSQL database URL from configuration
    
    Returns:
        str: Database URL string
    """
    config = get_database_config()
    return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
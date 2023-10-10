"""
File that contains various utils
"""

from pathlib import Path


def get_project_root() -> str:
    """
    Function to return the root dir of the project
    """
    return str(Path(__file__).parent.parent)

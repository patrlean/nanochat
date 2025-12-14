"""
SFTData task for loading custom SFT training conversations.
Loads from the sftdata.jsonl file in the project root.
"""

import os
from tasks.customjson import CustomJSON


class SFTData(CustomJSON):
    """
    Load conversations from the sftdata.jsonl file.
    This is a convenience wrapper around CustomJSON that automatically
    locates the sftdata.jsonl file in the project root.
    
    Usage:
        from tasks.sftdata import SFTData
        ds = SFTData()
    """

    def __init__(self, **kwargs):
        # Get the project root directory (nanochat/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        filepath = os.path.join(project_root, "sftdata_v1.jsonl")
        super().__init__(filepath=filepath, **kwargs)

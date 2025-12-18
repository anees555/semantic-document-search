"""
Document Loading and Chunking Module

Loads documents from formats (.txt, .md, .pdf)
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


# for pdf processing
try:
    from PyPDF2 import PdfReader
    pdf_support = True

except ImportError:
    pdf_support = False
    print("Pdf not supported")

@dataclass

class DocumentChunk:
    """Represents a single chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

class DocumentLoader:
    """Load and process document for vector storage"""
    def __init__(self, 
                 chunk_size:int = 500, 
                 chunk_overlap:int = 50, 
                 min_chunk_size:int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.supported_extension = ['.txt', '.md', '.pdf']

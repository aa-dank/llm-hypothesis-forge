# services/utils.py

import logging
import json
import os
import re
from typing import Dict, List, Optional, Any, Set, Union, Callable
from data.models import ResearchPaper
from data.utils import make_api_request
from io import BytesIO

logger = logging.getLogger(__name__)


def extract_structured_text(paper: ResearchPaper, components: Dict[str, str], max_text_length: int = 10000) -> str:
    """
    Format paper metadata into a structured text representation.
    
    Args:
        paper (ResearchPaper): The paper object containing metadata
        components (Dict[str, str]): Dictionary mapping component names to values
        max_text_length (int): Maximum length for any text component (to avoid overwhelming the system)
        
    Returns:
        str: Structured text representation of the paper
    """
    formatted_components = []
    
    # Process each component
    for name, value in components.items():
        if value:
            if name.lower() == "text" and len(value) > max_text_length:
                # Truncate very long text
                formatted_text = f"{value[:max_text_length]}... [truncated]"
                formatted_components.append(f"{name.title()}: {formatted_text}")
            else:
                formatted_components.append(f"{name.title()}: {value}")
    
    return "\n\n".join(formatted_components)


def get_fallback_text(paper: ResearchPaper) -> str:
    """
    Get fallback text for a paper when full text retrieval fails.
    
    Args:
        paper (ResearchPaper): The paper object
        
    Returns:
        str: Fallback text (usually the abstract)
    """
    return paper.abstract if paper.abstract else ""


def safe_api_request(url: str, description: str, headers: Optional[Dict] = None) -> Optional[Any]:
    """
    Make a safe API request with error handling.
    
    Args:
        url (str): The URL to request
        description (str): Description of the request for logging
        headers (Optional[Dict]): Optional headers for the request
        
    Returns:
        Optional[Any]: Response object or None if request failed
    """
    try:
        logger.info(f"Making API request: {description}")
        response = make_api_request(url, description, headers=headers)
        return response
    except Exception as e:
        logger.error(f"API request failed ({description}): {e}")
        return None


def load_api_credentials(env_vars: List[str]) -> Dict[str, str]:
    """
    Load API credentials from environment variables.
    
    Args:
        env_vars (List[str]): List of environment variable names to load
        
    Returns:
        Dict[str, str]: Dictionary mapping environment variable names to their values
    """
    from dotenv import load_dotenv
    
    load_dotenv()
    credentials = {}
    missing_vars = []
    
    for var in env_vars:
        value = os.getenv(var)
        credentials[var] = value
        if not value:
            missing_vars.append(var)
            
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        
    return credentials


def chunk_text(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of maximum size.
    
    Args:
        text (str): The text to chunk
        max_chunk_size (int): Maximum size of each chunk
        overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Take a chunk of max_chunk_size or the remaining text if shorter
        end = min(start + max_chunk_size, len(text))
        
        # If this isn't the first chunk and we're not at the end, try to find a good break point
        if start > 0 and end < len(text):
            # Find the last period, newline, or space in this chunk
            for break_char in ['. ', '.\n', '\n\n', '\n', ' ']:
                last_break = text[start:end].rfind(break_char)
                if last_break != -1:
                    end = start + last_break + len(break_char)
                    break
        
        chunks.append(text[start:end])
        
        # Move start point for next chunk, incorporating overlap
        start = max(start, end - overlap)
    
    return chunks


def create_generic_paper(
    doi: str = "",
    title: str = "",
    authors: str = "",
    date: str = "",
    abstract: str = "",
    category: str = "",
    journal: str = ""
) -> ResearchPaper:
    """
    Create a generic ResearchPaper object with the provided metadata.
    
    Args:
        doi (str): Digital Object Identifier
        title (str): Paper title
        authors (str): Paper authors (semicolon-separated)
        date (str): Paper publication date
        abstract (str): Paper abstract
        category (str): Paper category/field
        journal (str): Journal name
        
    Returns:
        ResearchPaper: A new ResearchPaper object
    """
    return ResearchPaper(
        doi=doi,
        title=title,
        authors=authors,
        date=date,
        abstract=abstract,
        category=category,
        license="",
        version="",
        author_corresponding="",
        author_corresponding_institution="",
        published_journal=journal,
        published_date=date,
        published_doi=doi,
        inclusion_decision=None,
        criteria_assessment=None,
        assessment_explanation=None,
        assessment_datetime=None
    )

def pdf_to_text(pdf_content, max_pages=None) -> Optional[str]:
    """
    Extract text from PDF bytes or BytesIO using pdfplumber.
    Returns the cleaned text, or None on error or if no text was found.
    """
    try:
        import pdfplumber
    except ImportError:
        msg = "PDF extraction requires pdfplumber. Install with 'pip install pdfplumber'"
        logger.error(msg)
        raise

    buf = BytesIO(pdf_content) if isinstance(pdf_content, (bytes, bytearray)) else pdf_content
    buf.seek(0)

    try:
        text_parts = []
        with pdfplumber.open(buf) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

        if not text_parts:
            logger.warning("PDF extraction succeeded but yielded no text")
            return None

        # collapse runs of ≥3 newlines to 2
        text = "\n\n".join(text_parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # strip only C0 controls, keep normal unicode
        text = re.sub(r"[\x00-\x1F]+", "", text).strip()

        return text or None

    except Exception:
        logger.exception("Error extracting text from PDF")
        return None
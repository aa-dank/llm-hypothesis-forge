import logging
import os
from typing import Optional
from data.utils import make_api_request
from data.models import ResearchPaper
from services.utils import pdf_to_text

logger = logging.getLogger(__name__)

UNPAYWALL_API_URL = "https://api.unpaywall.org/v2/"

def get_unpaywall_metadata(doi: str, mailto: str) -> Optional[dict]:
    """
    Fetch metadata from the Unpaywall API for a given DOI.
    
    Args:
        doi (str): The DOI of the research paper.
        mailto (str): A valid email address for API identification.
        
    Returns:
        Optional[dict]: The metadata dictionary if successful, else None.
    """
    url = f"{UNPAYWALL_API_URL}{doi}?email={mailto}"
    response = make_api_request(url, f"Unpaywall metadata for DOI {doi}")
    
    if not response or response.status_code != 200:
        logger.warning(f"Unpaywall metadata fetch failed for DOI {doi}")
        return None
    
    try:
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to parse Unpaywall response for DOI {doi}: {e}")
        return None

def fetch_full_text(paper: ResearchPaper) -> str:
    """
    Attempt to fetch the full text of a research paper via Unpaywall.

    Args:
        paper (ResearchPaper): The paper object containing the DOI.

    Returns:
        str: Full text content or abstract if retrieval fails.
    """
    if not paper.doi:
        logger.warning("No DOI provided for Unpaywall paper")
        return paper.abstract if paper.abstract else ""
        
    # Get email from environment or use default
    mailto = os.getenv('UNPAYWALL_EMAIL', 'project@example.com')
    
    logger.info(f"Fetching Unpaywall data for DOI: {paper.doi}")
    metadata = get_unpaywall_metadata(paper.doi, mailto)
    
    if not metadata:
        logger.warning(f"No Unpaywall metadata found for DOI: {paper.doi}")
        return paper.abstract if paper.abstract else ""
    
    # Check if paper is open access
    is_oa = metadata.get('is_oa', False)
    
    if not is_oa:
        logger.info(f"Paper with DOI {paper.doi} is not open access according to Unpaywall")
        return _format_metadata_text(metadata, paper)
    
    # Try to get full text from best_oa_location first
    best_location = metadata.get("best_oa_location")
    if best_location and best_location.get("url_for_pdf"):
        pdf_url = best_location.get("url_for_pdf")
        if pdf_url:
            logger.info(f"Attempting to download PDF from {pdf_url}")
            pdf_text = _download_and_extract_pdf(pdf_url, paper.doi)
            if pdf_text:
                return _format_full_text(metadata, pdf_text, paper)
    
    # If best_oa_location didn't work, try other oa_locations
    oa_locations = metadata.get("oa_locations", [])
    for location in oa_locations:
        pdf_url = location.get("url_for_pdf")
        if pdf_url:
            logger.info(f"Attempting to download PDF from alternate location: {pdf_url}")
            pdf_text = _download_and_extract_pdf(pdf_url, paper.doi)
            if pdf_text:
                return _format_full_text(metadata, pdf_text, paper)
    
    logger.warning(f"No accessible full text found for DOI {paper.doi}")
    return _format_metadata_text(metadata, paper)

def _download_and_extract_pdf(url: str, doi: str) -> Optional[str]:
    """
    Download a PDF from a URL and extract its text content.
    
    Args:
        url (str): URL to fetch the PDF from.
        doi (str): DOI for logging purposes.
        
    Returns:
        Optional[str]: Extracted text or None if extraction fails.
    """
    try:
        response = make_api_request(url, f"PDF from {url} for DOI {doi}")
        
        if not response or response.status_code != 200 or not response.content:
            return None
            
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type:
            logger.warning(f"URL does not point to a PDF: {url}, content-type: {content_type}")
            return None
            
        return pdf_to_text(response.content)
    except Exception as e:
        logger.error(f"Error downloading/extracting PDF from {url}: {e}")
        return None

def _format_full_text(metadata: dict, pdf_text: str, paper: ResearchPaper) -> str:
    """
    Format extracted PDF text with metadata.
    
    Args:
        metadata (dict): Unpaywall metadata.
        pdf_text (str): Extracted text from PDF.
        paper (ResearchPaper): Original paper object.
        
    Returns:
        str: Formatted text combining metadata and full text.
    """
    components = []
    
    # Add title
    title = metadata.get('title') or paper.title
    if title:
        components.append(f"Title: {title}")
        
    # Add authors
    if paper.authors:
        components.append(f"Authors: {paper.authors}")
        
    # Add journal information
    journal = metadata.get('journal_name')
    if journal:
        components.append(f"Journal: {journal}")
        
    # Add DOI
    if paper.doi:
        components.append(f"DOI: {paper.doi}")
        
    # Add full text
    components.append(f"Full Text:\n\n{pdf_text}")
    
    return "\n\n".join(components)

def _format_metadata_text(metadata: dict, paper: ResearchPaper) -> str:
    """
    Format metadata into readable text when full text isn't available.
    
    Args:
        metadata (dict): Unpaywall metadata.
        paper (ResearchPaper): Original paper object.
        
    Returns:
        str: Formatted metadata text.
    """
    components = []
    
    # Add title
    title = metadata.get('title') or paper.title
    if title:
        components.append(f"Title: {title}")
    
    # Add abstract from paper if available
    if paper.abstract:
        components.append(f"Abstract: {paper.abstract}")
        
    # Add authors from paper
    if paper.authors:
        components.append(f"Authors: {paper.authors}")
        
    # Add journal information
    journal = metadata.get('journal_name')
    if journal:
        components.append(f"Journal: {journal}")
        
    # Add publication year
    year = metadata.get('year')
    if year:
        components.append(f"Year: {year}")
        
    # Add publisher
    publisher = metadata.get('publisher')
    if publisher:
        components.append(f"Publisher: {publisher}")
        
    # Add DOI
    if paper.doi:
        components.append(f"DOI: {paper.doi}")
    
    # Add open access status
    oa_status = metadata.get('oa_status')
    if oa_status:
        components.append(f"Open Access Status: {oa_status}")
        
    return "\n\n".join(components)
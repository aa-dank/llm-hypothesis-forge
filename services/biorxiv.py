import logging
from data.utils import make_api_request
from data.models import ResearchPaper
from services.service_model import ServiceQueryInfo
from services.utils import pdf_to_text

logger = logging.getLogger(__name__)

service_info = ServiceQueryInfo(
    name="bioRxiv",
    description="A free online archive and distribution service for unpublished preprints in the life sciences.",
    domains=["biology", "neuroscience", "genetics", "bioinformatics", "immunology", "molecular biology", "cell biology", "ecology", "evolutionary biology"],
    query_format="boolean",
    search_fields=["title", "abstract", "full text"],
    enabled=True
)

def fetch_biorxiv_details(interval, cursor, output_format):
    """Fetch paper details from bioRxiv API."""
    details_base_url = 'https://api.biorxiv.org/details/biorxiv/{}/{}/{}'
    details_url = details_base_url.format(interval, cursor, output_format)
    return make_api_request(details_url, "bioRxiv details")

def fetch_biorxiv_publication_info(doi, output_format):
    """Fetch publication info for a bioRxiv paper."""
    pubs_base_url = 'https://api.biorxiv.org/pubs/biorxiv/{}/na/{}'
    pubs_url = pubs_base_url.format(doi, output_format)
    print(f"  Fetching publication info for DOI: {doi}")
    return make_api_request(pubs_url, f"publication info for DOI {doi}")

def create_biorxiv_paper(record, publication_info=None):
    """Create a ResearchPaper object from bioRxiv API data."""
    paper = ResearchPaper(
        doi=record.get("doi", ""),
        title=record.get("title", ""),
        authors=record.get("authors", ""),
        date=record.get("date", ""),
        abstract=record.get("abstract", ""),
        category=record.get("category", ""),
        license=record.get("license", ""),
        version=record.get("version", ""),
        author_corresponding=record.get("author_corresponding", ""),
        author_corresponding_institution=record.get("author_corresponding_institution", ""),
        inclusion_decision=None,
        criteria_assessment=None,
        assessment_explanation=None,
        assessment_datetime=None  # Explicitly set to None to ensure it's blank
    )
    
    # Add publication info if available
    if publication_info:
        paper.published_journal = publication_info.get("published_journal", "")
        paper.published_date = publication_info.get("published_date", "")
        paper.published_doi = publication_info.get("published_doi", "")
    
    return paper

def fetch_biorxiv_paper_by_doi(doi):
    """
    Fetch paper details from bioRxiv API by DOI and create a ResearchPaper object.
    
    Args:
        doi (str): The bioRxiv/medRxiv DOI
    
    Returns:
        ResearchPaper or None: A ResearchPaper object with the paper metadata, or None if retrieval fails
    """
    try:
        # Make a request to the bioRxiv API
        details_url = f'https://api.biorxiv.org/details/biorxiv/{doi}/na/json'
        details_response = make_api_request(details_url, f"bioRxiv details for DOI {doi}")
        if not details_response:
            return None
        
        # Parse the response
        details_data = details_response.json()
        records = details_data.get('collection', [])
        
        if not records:
            print(f"  No data found for bioRxiv DOI {doi}")
            return None
        
        # Create a ResearchPaper object from the first record
        paper = create_biorxiv_paper(records[0])
        
        # Fetch and add publication info
        pubs_url = f'https://api.biorxiv.org/pubs/biorxiv/{doi}/na/json'
        pubs_response = make_api_request(pubs_url, f"publication info for DOI {doi}")
        if pubs_response:
            process_publication_info(pubs_response, paper)
        
        return paper
        
    except Exception as e:
        print(f"  Error fetching paper from bioRxiv: {e}")
        return None
    
def process_publication_info(response, paper):
    """Process the publication info response and update the paper object."""
    if not response:
        return
    
    pubs_data = response.json()
    pubs_collection = pubs_data.get("collection", [])
    
    if pubs_collection:
        pub_info = pubs_collection[0]
        paper.published_journal = pub_info.get("published_journal", "")
        paper.published_date = pub_info.get("published_date", "")
        paper.published_doi = pub_info.get("published_doi", "")
    else:
        print(f"    No published info found for DOI {paper.doi}.")

def import_biorxiv_paper(doi, session):
    """
    Fetches metadata for a biorXiv paper and creates a ResearchPaper object.
    """
    details_url_template = 'https://api.biorxiv.org/details/biorxiv/{}/na/json'
    pubs_url_template = 'https://api.biorxiv.org/pubs/biorxiv/{}/na/json'
    
    try:
        # Fetch details from biorxiv API
        details_response = make_api_request(
            details_url_template.format(doi),
            f"bioRxiv details for DOI {doi}"
        )
        if not details_response:
            return None
            
        details_data = details_response.json()
        records = details_data.get('collection', [])
        
        if not records:
            print(f"No details found for DOI {doi}")
            return None
        
        record = records[0]  # Take the first record (usually the most recent version)
        paper = create_biorxiv_paper(record)
        paper.previously_in_brainbench = True
        
        # Fetch publication info
        pubs_response = make_api_request(
            pubs_url_template.format(doi),
            f"publication info for DOI {doi}"
        )
        if pubs_response:
            process_publication_info(pubs_response, paper)
        
        return paper
        
    except Exception as e:
        print(f"Error fetching metadata for biorXiv DOI {doi}: {e}")
        return None

def run_queries(queries: list[str], max_results: int = 10) -> list[ResearchPaper]:
    """
    Execute each query against the bioRxiv service and return a flat list of ResearchPaper objects.
    
    Args:
        queries: List of search queries to execute against bioRxiv
        max_results: Maximum number of results to return per query
        
    Returns:
        List of ResearchPaper objects from query results
    """
    papers = []
    # bioRxiv API doesn't directly support searching by keywords
    # We use a timeframe-based approach with the most recent papers
    
    # Get last 7 days of papers
    interval = "7d"  # Last 7 days
    cursor = 0
    output_format = "json"
    
    for q in queries:
        # For now, we fetch recent papers and filter them manually
        # Note: The actual bioRxiv API lacks a direct search endpoint for keywords
        # In a production environment, consider using a better search approach
        response = fetch_biorxiv_details(interval, cursor, output_format)
        if not response:
            continue
        
        try:
            data = response.json()
            records = data.get('collection', [])
            
            # Simple filtering by checking if query terms appear in title/abstract
            query_terms = q.lower().split()
            matched_papers = []
            
            for record in records:
                title = record.get("title", "").lower()
                abstract = record.get("abstract", "").lower()
                text = title + " " + abstract
                
                # Check if all query terms appear somewhere in the text
                if all(term in text for term in query_terms):
                    paper = create_biorxiv_paper(record)
                    matched_papers.append(paper)
                
                # Break once we have enough papers
                if len(matched_papers) >= max_results:
                    break
            
            papers.extend(matched_papers[:max_results])
            
        except Exception as e:
            print(f"Error processing bioRxiv results for query '{q}': {e}")
    
    return papers

def fetch_full_text(paper: ResearchPaper) -> str:
    """
    Fetches the full text or enhanced metadata for a bioRxiv paper.
    
    Args:
        paper (ResearchPaper): A ResearchPaper object with bioRxiv DOI
        
    Returns:
        str: The full text of the paper if available, or enhanced metadata if PDF retrieval fails
    """
    logger.info(f"Fetching full text for bioRxiv paper with DOI: {paper.doi}")
    
    # First check if paper has an abstract we can return as fallback
    if not paper.doi:
        logger.warning("No DOI provided for bioRxiv paper")
        return paper.abstract if paper.abstract else ""
    
    try:
        # Make sure the DOI is a bioRxiv DOI
        if not paper.doi.startswith("10.1101/"):
            logger.warning(f"DOI {paper.doi} does not appear to be a bioRxiv DOI")
            return paper.abstract if paper.abstract else ""
            
        # First attempt: retrieve the actual PDF
        # Construct the PDF URL (format: https://www.biorxiv.org/content/{doi}v{version}.full.pdf)
        # If we don't know the version, try the latest (default)
        version = paper.version if paper.version else "1"
        pdf_url = f"https://www.biorxiv.org/content/{paper.doi}v{version}.full.pdf"
        
        logger.info(f"Attempting to download PDF from {pdf_url}")
        
        try:
            # Download the PDF using make_api_request instead of requests.get
            response = make_api_request(pdf_url, f"PDF for DOI {paper.doi}")
            if response and response.status_code == 200 and response.content:
                # Extract text from the PDF using the utility function from services.utils
                pdf_text = pdf_to_text(response.content)
                
                if pdf_text:
                    logger.info(f"Successfully extracted {len(pdf_text)} characters from PDF for DOI: {paper.doi}")
                    
                    # Use the bioRxiv API to fetch detailed information for metadata
                    details_url = f'https://api.biorxiv.org/details/biorxiv/{paper.doi}/na/json'
                    details_response = make_api_request(details_url, f"bioRxiv details for DOI {paper.doi}")
                    
                    # Prepare components for structured text output
                    components = []
                    
                    # Add basic metadata from the paper object if API fails
                    if not details_response:
                        components.append(f"Title: {paper.title}" if paper.title else "")
                        components.append(f"Authors: {paper.authors}" if paper.authors else "")
                        components.append(f"Date: {paper.date}" if paper.date else "")
                        components.append(f"DOI: {paper.doi}")
                    else:
                        # Parse the response
                        details_data = details_response.json()
                        records = details_data.get('collection', [])
                        
                        if records:
                            record = records[0]
                            components.append(f"Title: {record.get('title')}" if record.get('title') else "")
                            components.append(f"Authors: {record.get('authors')}" if record.get('authors') else "")
                            components.append(f"Date: {record.get('date')}" if record.get('date') else "")
                            components.append(f"Version: {record.get('version')}" if record.get('version') else "")
                            components.append(f"Category: {record.get('category')}" if record.get('category') else "")
                            components.append(f"DOI: {paper.doi}")
                    
                    # Add publication information if available
                    pubs_url = f'https://api.biorxiv.org/pubs/biorxiv/{paper.doi}/na/json'
                    pubs_response = make_api_request(pubs_url, f"publication info for DOI {paper.doi}")
                    
                    if pubs_response:
                        pubs_data = pubs_response.json()
                        pubs_collection = pubs_data.get("collection", [])
                        if pubs_collection:
                            pub_info = pubs_collection[0]
                            if pub_info.get("published_journal"):
                                components.append(f"Published in: {pub_info.get('published_journal')}")
                                
                            if pub_info.get("published_date"):
                                components.append(f"Publication Date: {pub_info.get('published_date')}")
                                
                            if pub_info.get("published_doi"):
                                components.append(f"Publication DOI: {pub_info.get('published_doi')}")
                    
                    # Add the full text
                    components.append(f"Full Text:\n\n{pdf_text}")
                    
                    # Combine all components
                    return "\n\n".join([c for c in components if c])
        
        except ImportError:
            logger.warning("PDF extraction libraries not available. Falling back to metadata.")
        except Exception as e:
            logger.warning(f"Error downloading or processing PDF: {e}. Falling back to metadata.")
        
        # Fallback: If we couldn't get the PDF, fetch enhanced metadata
        logger.info(f"Falling back to metadata for DOI: {paper.doi}")
        
        # Use the bioRxiv API to fetch detailed information
        details_url = f'https://api.biorxiv.org/details/biorxiv/{paper.doi}/na/json'
        details_response = make_api_request(details_url, f"bioRxiv details for DOI {paper.doi}")
        
        if not details_response:
            logger.warning(f"Could not fetch details for bioRxiv paper with DOI: {paper.doi}")
            return paper.abstract if paper.abstract else ""
            
        # Parse the response
        details_data = details_response.json()
        records = details_data.get('collection', [])
        
        if not records:
            logger.warning(f"No data found for bioRxiv DOI {paper.doi}")
            return paper.abstract if paper.abstract else ""
            
        # Get the first record (most recent version)
        record = records[0]
        
        # Also fetch publication information if available
        pubs_url = f'https://api.biorxiv.org/pubs/biorxiv/{paper.doi}/na/json'
        pubs_response = make_api_request(pubs_url, f"publication info for DOI {paper.doi}")
        pub_info = {}
        
        if pubs_response:
            pubs_data = pubs_response.json()
            pubs_collection = pubs_data.get("collection", [])
            if pubs_collection:
                pub_info = pubs_collection[0]
        
        # Compile all available information into a comprehensive text
        components = []
        
        # Add title
        if record.get("title"):
            components.append(f"Title: {record.get('title')}")
        
        # Add abstract
        if record.get("abstract"):
            components.append(f"Abstract: {record.get('abstract')}")
        
        # Add authors
        if record.get("authors"):
            components.append(f"Authors: {record.get('authors')}")
        
        # Add dates
        if record.get("date"):
            components.append(f"Preprint Date: {record.get('date')}")
            
        # Add version
        if record.get("version"):
            components.append(f"Version: {record.get('version')}")
            
        # Add category
        if record.get("category"):
            components.append(f"Category: {record.get('category')}")
            
        # Add license
        if record.get("license"):
            components.append(f"License: {record.get('license')}")
        
        # Add corresponding author details
        if record.get("author_corresponding"):
            components.append(f"Corresponding Author: {record.get('author_corresponding')}")
            
        if record.get("author_corresponding_institution"):
            components.append(f"Corresponding Institution: {record.get('author_corresponding_institution')}")
        
        # Add publication information if available
        if pub_info:
            if pub_info.get("published_journal"):
                components.append(f"Published in: {pub_info.get('published_journal')}")
                
            if pub_info.get("published_date"):
                components.append(f"Publication Date: {pub_info.get('published_date')}")
                
            if pub_info.get("published_doi"):
                components.append(f"Publication DOI: {pub_info.get('published_doi')}")
        
        # Combine all components
        enhanced_text = "\n\n".join(components)
        logger.info(f"Successfully compiled enhanced information for bioRxiv DOI {paper.doi}")
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Error fetching full text for bioRxiv paper: {e}")
        return paper.abstract if paper.abstract else ""

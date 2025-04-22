import httpx
import logging
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
from services.service_model import ServiceQueryInfo
from data.models import ResearchPaper

logger = logging.getLogger(__name__)

service_info = ServiceQueryInfo(
    name="PubMed",
    description="A comprehensive database of biomedical literature, including life sciences, behavioral sciences, chemical sciences, and bioengineering.",
    domains=["neuroscience", "medicine", "biology", "optometry"],
    query_format="keyword",
    search_fields=["title", "abstract"],
    enabled=True
)

def _pubmed_request(base_url, params):
    """Helper function to handle PubMed API requests with retries."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(base_url, params=params)
            time.sleep(0.35)  # Adhere to NCBI E-utils guidelines (max 3 requests/sec without API key)
            
            if response.status_code == 429:
                # Rate limit reached, wait and retry
                time.sleep(2)
                response = client.get(base_url, params=params)  # Retry once
                
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response
    except httpx.RequestError as e:
        print(f"PubMed API request failed: {e}")
        return None

def get_pmid_from_doi(doi):
    """
    Search PubMed for a PMID using a DOI.
    Inputs:
        doi: str, Digital Object Identifier
    Outputs:
        str: PMID if found, else None
    """
    if not doi:
        return None
        
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]",  # Search specifically in the DOI field
        "retmax": 1,
        "retmode": "xml"
    }
    
    response = _pubmed_request(base_url, params)
    if response:
        try:
            root = ET.fromstring(response.content)
            id_elem = root.find(".//Id")
            if id_elem is not None and id_elem.text:
                print(f"Found PMID {id_elem.text} for DOI {doi}")
                return id_elem.text
            else:
                print(f"No PMID found for DOI {doi}")
                return None
        except ET.ParseError as e:
            print(f"Error parsing PubMed search response for DOI {doi}: {e}")
            return None
    return None

def search_pubmed(query, max_results=100, date_range_years=1):
    """
    Search PubMed for articles matching query given a date range
    Inputs:
        query: str, keywords to search for
        max_results: int, max number of results
        date_range_years: int, number of years to search back
    Outputs:
        list of PMIDs
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*date_range_years)
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml",
        "datetype": "pdat",
        "mindate": start_date.strftime("%Y/%m/%d"),
        "maxdate": end_date.strftime("%Y/%m/%d")
    }

    response = _pubmed_request(base_url, params)
    if response:
        try:
            root = ET.fromstring(response.content)
            id_list = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
            return id_list
        except ET.ParseError as e:
            print(f"Error parsing PubMed search response: {e}")
            return []
    return []

def extract_paper_features(pmid):
    """
    Extract features from a PubMed paper using its PMID.
    Inputs:
        pmid: str, PubMed ID
    Outputs:
        dict of features or None if fetching fails or article not found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    
    response = _pubmed_request(base_url, params)
    if not response:
        return None

    try:
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")  # Look for PubmedArticle element
        if not article:
            article = root.find(".//Article")  # Fallback for some structures
            
        if not article:
            print(f"No Article found in XML for PMID {pmid}")
            return None
            
        # --- Extract Abstract ---
        abstract_texts = article.findall(".//Abstract/AbstractText")
        full_abstract = ""
        if abstract_texts:
            # Combine sections, adding labels if present
            parts = []
            for part in abstract_texts:
                label = part.get('Label')
                text = part.text if part.text else ""
                if label and text:
                    parts.append(f"{label.upper()}: {text.strip()}")
                elif text:
                    parts.append(text.strip())
            full_abstract = "\n".join(parts)
        else:
             # Sometimes abstract is directly under Abstract tag
             abstract_node = article.find(".//Abstract")
             if abstract_node is not None and abstract_node.text:
                 full_abstract = abstract_node.text.strip()

        if not full_abstract:
            print(f"No abstract text found for PMID {pmid}")
            # Decide if you want to return None or continue without abstract
            # return None 

        # --- Extract Authors ---
        authors_list = []
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            firstname = author.find("ForeName")  # Sometimes ForeName, sometimes FirstName
            if firstname is None:
                firstname = author.find("FirstName")
            
            author_str = ""
            if lastname is not None and lastname.text:
                author_str += lastname.text
            if firstname is not None and firstname.text:
                if author_str: author_str += ", "
                author_str += firstname.text
            if author_str:
                authors_list.append(author_str)
        authors_str = "; ".join(authors_list)  # Format as semicolon-separated string

        # --- Extract Publication Date ---
        pub_date_str = ""
        pub_date_node = article.find(".//Journal//PubDate")  # Common location
        if pub_date_node is not None:
            year = pub_date_node.find("Year")
            month = pub_date_node.find("Month")
            day = pub_date_node.find("Day")
            
            # Attempt to build YYYY-MM-DD, fallback to YYYY-MM or YYYY
            try:
                date_parts = []
                if year is not None and year.text: date_parts.append(year.text)
                if month is not None and month.text: date_parts.append(month.text.zfill(2))  # Pad month
                if day is not None and day.text: date_parts.append(day.text.zfill(2))  # Pad day
                pub_date_str = "-".join(date_parts)
            except Exception:  # Handle potential errors if month isn't numeric etc.
                 if year is not None and year.text: pub_date_str = year.text


        # --- Extract Other Fields ---
        title_node = article.find(".//ArticleTitle")
        journal_node = article.find(".//Journal/Title")
        doi_node = article.find('.//ArticleId[@IdType="doi"]')  # Find DOI if present

        features = {
            "pmid": pmid,
            "doi": doi_node.text if doi_node is not None else None,
            "title": title_node.text.strip() if title_node is not None and title_node.text else None,
            "full_abstract": full_abstract if full_abstract else None,
            "journal": journal_node.text.strip() if journal_node is not None and journal_node.text else None,
            "pub_date_str": pub_date_str if pub_date_str else None,
            "authors_str": authors_str if authors_str else None,
            # Keep structured abstract if needed, though full_abstract is the main goal now
            # "abstract_sections": abstract_sections, 
        }
        
        # Check for missing essential fields
        if not features["title"]: print(f"Missing title for PMID {pmid}")
        if not features["full_abstract"]: print(f"Missing abstract for PMID {pmid}")
            
        return features

    except ET.ParseError as e:
        print(f"Error parsing PubMed fetch response for PMID {pmid}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing paper {pmid}: {e}")
        return None

def fetch_full_text(paper: ResearchPaper) -> str:
    """
    Fetches the full text or enhanced metadata for a PubMed paper.
    
    Args:
        paper (ResearchPaper): A ResearchPaper object with PubMed ID or DOI
        
    Returns:
        str: The full text or enhanced metadata of the paper if available, 
             or the abstract if full text cannot be retrieved
    """
    logger.info(f"Fetching full text for PubMed paper with DOI: {paper.doi}")
    
    try:
        # Try to get PMID, either directly from the paper or by looking it up
        pmid = None
        
        # If the paper has a PMID directly
        if hasattr(paper, 'pmid') and paper.pmid:
            pmid = paper.pmid
        # Otherwise, try to get PMID from DOI
        elif paper.doi:
            pmid = get_pmid_from_doi(paper.doi)
            
        if not pmid:
            logger.warning(f"Could not determine PMID for paper with DOI: {paper.doi}")
            return paper.abstract if paper.abstract else ""
            
        # Now that we have a PMID, fetch the paper details
        features = extract_paper_features(pmid)
        
        if not features:
            logger.warning(f"Could not extract features for PMID: {pmid}")
            return paper.abstract if paper.abstract else ""
            
        # Compile all available information into a comprehensive text
        components = []
        
        # Add title
        if features.get("title"):
            components.append(f"Title: {features.get('title')}")
        
        # Add abstract
        if features.get("full_abstract"):
            components.append(f"Abstract: {features.get('full_abstract')}")
        
        # Add authors
        if features.get("authors_str"):
            components.append(f"Authors: {features.get('authors_str')}")
        
        # Add journal information
        if features.get("journal"):
            components.append(f"Journal: {features.get('journal')}")
            
        # Add publication date
        if features.get("pub_date_str"):
            components.append(f"Publication Date: {features.get('pub_date_str')}")
            
        # Add DOI
        if features.get("doi"):
            components.append(f"DOI: {features.get('doi')}")
            
        # Add PMID reference
        components.append(f"PMID: {pmid}")
        
        # Combine all components
        enhanced_text = "\n\n".join(components)
        
        logger.info(f"Successfully compiled enhanced information for PMID {pmid}")
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Error fetching full text for PubMed paper: {e}")
        return paper.abstract if paper.abstract else ""

def run_queries(queries: list[str], max_results: int = 10) -> list[ResearchPaper]:
    """
    Execute each query against the PubMed service and return a flat list of ResearchPaper objects.
    
    Args:
        queries: List of search queries to execute against PubMed
        max_results: Maximum number of results to return per query
        
    Returns:
        List of ResearchPaper objects from query results
    """
    papers = []
    
    for q in queries:
        # Search PubMed for matching papers
        pmids = search_pubmed(query=q, max_results=max_results)
        
        # For each PMID, fetch the paper details
        for pmid in pmids:
            features = extract_paper_features(pmid)
            if features:
                # Convert to ResearchPaper object
                paper = ResearchPaper(
                    doi=features.get('doi', ''),
                    title=features.get('title', ''),
                    authors=features.get('authors_str', ''),
                    date=features.get('pub_date_str', ''),
                    abstract=features.get('full_abstract', ''),
                    category='',  # PubMed doesn't provide category information in this format
                    license='',  # PubMed doesn't provide license information
                    version='',  # Not applicable for published papers
                    author_corresponding='',  # Not directly available from basic PubMed data
                    author_corresponding_institution='',  # Not directly available from basic PubMed data
                    published_journal=features.get('journal', ''),
                    published_date=features.get('pub_date_str', ''),
                    published_doi=features.get('doi', ''),
                    inclusion_decision=None,
                    criteria_assessment=None,
                    assessment_explanation=None,
                    assessment_datetime=None  # Explicitly set to None to ensure it's blank
                )
                papers.append(paper)
    
    return papers
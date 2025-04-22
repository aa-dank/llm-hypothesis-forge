import pyalex
from pyalex import Works
from services.service_model import ServiceQueryInfo
from data.models import ResearchPaper
import logging

logger = logging.getLogger(__name__)

service_info = ServiceQueryInfo(
    name="OpenAlex",
    description="An open catalog of the global research system, indexing scholarly works, authors, venues, institutions, and concepts across various disciplines.",
    domains=["neuroscience", "medicine", "biology", "optometry", "computer science", "economics"],
    query_format="natural",
    search_fields=["title", "abstract", "full text"],
    enabled=True
)

def fetch_openalex_paper_by_doi(doi: str, mailto: str = None):
    """
    Retrieve paper metadata from OpenAlex given a DOI.
    
    The function normalizes the DOI if it is passed as a URL 
    (e.g. "https://doi.org/..."), sets a contact email for polite use
    of the OpenAlex API (if provided), and converts the DOI into the
    identifier format expected by OpenAlex, namely "doi:<DOI>".
    
    Args:
        doi (str): The paper's DOI which might be a plain DOI string
                   (e.g. "10.1038/nphys1170") or a URL (e.g. "https://doi.org/10.1038/nphys1170").
        mailto (str, optional): An email address to include in OpenAlex's requests.
    
    Returns:
        dict or None: The metadata for the paper as returned by OpenAlex via PyAlex,
                      or None if an error occurs.
    """
    # Normalize the DOI by stripping any URL prefix
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    elif doi.startswith("http://doi.org/"):
        doi = doi.replace("http://doi.org/", "")
    
    # If an email is provided, set it in the PyAlex configuration for rate-limiting purposes
    if mailto:
        pyalex.config.email = mailto

    # Construct the identifier expected by OpenAlex (i.e., "doi:<DOI>")
    identifier = f"doi:{doi}"

    try:
        # Retrieve the work using PyAlex; this call uses HTTPX internally if necessary.
        work = Works()[identifier]
        return work
    except Exception as e:
        logger.error(f"Error retrieving OpenAlex work for DOI {doi}: {e}")
        return None

def run_queries(queries: list[str], max_results: int = 10) -> list[ResearchPaper]:
    """
    Execute each query against the OpenAlex service and return a flat list of ResearchPaper objects.
    
    Args:
        queries: List of search queries to execute against OpenAlex
        max_results: Maximum number of results to return per query
        
    Returns:
        List of ResearchPaper objects from query results
    """
    # Set up polite pool usage
    pyalex.config.email = "user@example.com"  # Replace with a real email in production
    papers = []
    
    for q in queries:
        logger.info(f"Searching OpenAlex with query: {q}")
        matching_papers = []
        
        try:
            # Search OpenAlex using the query
            works = Works().search(q).limit(max_results)
            
            for work in works:
                try:
                    # Extract basic metadata
                    doi = work.get('doi', '')
                    title = work.get('title', '')
                    pub_date = work.get('publication_date', '')
                    
                    # Extract authors
                    authors_list = []
                    if 'authorships' in work:
                        for authorship in work['authorships']:
                            if 'author' in authorship and 'display_name' in authorship['author']:
                                authors_list.append(authorship['author']['display_name'])
                    authors = "; ".join(authors_list)
                    
                    # Extract corresponding author info
                    corresponding_author = ""
                    corresponding_institution = ""
                    if 'authorships' in work and work['authorships']:
                        first_authorship = work['authorships'][0]
                        if 'author' in first_authorship and 'display_name' in first_authorship['author']:
                            corresponding_author = first_authorship['author']['display_name']
                        if 'institutions' in first_authorship and first_authorship['institutions']:
                            if 'display_name' in first_authorship['institutions'][0]:
                                corresponding_institution = first_authorship['institutions'][0]['display_name']
                    
                    # Extract abstract
                    abstract = ""
                    if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
                        try:
                            inverted_index = work['abstract_inverted_index']
                            words = []
                            for word, positions in inverted_index.items():
                                for pos in positions:
                                    while len(words) <= pos:
                                        words.append("")
                                    words[pos] = word
                            abstract = " ".join(words)
                        except Exception as e:
                            logger.error(f"Error reconstructing abstract: {e}")
                    
                    # Extract journal
                    journal_name = ""
                    if 'host_venue' in work and work['host_venue'] and 'display_name' in work['host_venue']:
                        journal_name = work['host_venue']['display_name']
                    
                    # Extract category
                    category = ""
                    if 'concepts' in work and work['concepts']:
                        concepts = sorted(work['concepts'], key=lambda x: x.get('score', 0), reverse=True)
                        if concepts and 'display_name' in concepts[0]:
                            category = concepts[0]['display_name']
                    
                    paper = ResearchPaper(
                        doi=doi,
                        title=title,
                        authors=authors,
                        date=pub_date,
                        abstract=abstract,
                        category=category,
                        license="",
                        version="",
                        author_corresponding=corresponding_author,
                        author_corresponding_institution=corresponding_institution,
                        published_journal=journal_name,
                        published_date=pub_date,
                        published_doi=doi,
                        inclusion_decision=None,
                        criteria_assessment=None,
                        assessment_explanation=None,
                        assessment_datetime=None
                    )
                    
                    matching_papers.append(paper)
                except Exception as e:
                    logger.error(f"Error processing OpenAlex work: {e}")
            
            papers.extend(matching_papers)
            logger.info(f"Found {len(matching_papers)} papers from OpenAlex for query: {q}")
        except Exception as e:
            logger.error(f"Error during OpenAlex search for query '{q}': {e}")
    
    return papers
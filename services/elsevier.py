import os
import json
import httpx
import logging
import unicodedata
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from urllib.parse import quote
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert
from services.service_model import ServiceQueryInfo
from data.models import ResearchPaper
from data.db import get_db_session, close_db_session

logger = logging.getLogger(__name__)

service_info = ServiceQueryInfo(
    name="Elsevier",
    description="A comprehensive abstract and citation database of peer-reviewed literature across various disciplines.",
    domains=["neuroscience", "medicine", "biology", "optometry", "computer science", "economics"],
    query_format="boolean",
    search_fields=["title", "abstract", "keywords"],
    enabled=True
)

def search_articles(query, api_key, insttoken):
    """
    Search for articles in Elsevier ScienceDirect using the provided query.
    
    Args:
        query (str): The search query to use for finding articles
        api_key (str): Elsevier API key for authentication
        insttoken (str): Elsevier institution token for authentication
        
    Returns:
        list: A list of DOIs for articles that match the query and have full text available
    """
    base_url = "https://api.elsevier.com/content/search/sciencedirect"
    headers = {
        "X-ELS-APIKey": api_key,
        "X-ELS-Insttoken": insttoken,
        "Accept": "application/json"
    }
    params = {
        "query": query,
        "count": 100
    }
    
    try:
        logger.info(f"Searching Elsevier articles with query: {query}")
        response = httpx.get(
            base_url,
            headers=headers,
            params=params
        )
        response.raise_for_status()
        results = response.json()
        
        dois = []
        
        if 'entry' not in results.get('search-results', {}):
            logger.warning("No articles found matching the criteria")
            return []
            
        for entry in results['search-results']['entry']:
            doi = entry.get('prism:doi')
            links = entry.get('link', [])
            has_full_text = any(link.get('@ref') in ['full-text', 'self'] for link in links)
            
            if doi and has_full_text:
                dois.append(doi)
        
        logger.info(f"Found {len(dois)} articles with full text access")
        return dois
            
    except Exception as e:
        logger.error(f"Error during Elsevier search: {e}")
        return []

def get_full_text_article(doi, api_key, insttoken): 
    """
    Retrieve full text article data from Elsevier API for a specific DOI.
    
    Args:
        doi (str): Digital Object Identifier for the article
        api_key (str): Elsevier API key for authentication
        insttoken (str): Elsevier institution token for authentication
        
    Returns:
        dict or None: JSON response containing article data if successful, None otherwise
    """
    headers = {
        "X-ELS-APIKey": api_key,
        "X-ELS-Insttoken": insttoken,
        "Accept": "application/json"
    }
    
    timeout = httpx.Timeout(10.0, connect=60.0)
    client = httpx.Client(timeout=timeout, headers=headers)
    url = f"https://api.elsevier.com/content/article/doi/{quote(doi)}"
    
    try:
        logger.info(f"Fetching full text for DOI: {doi}")
        r = client.get(url)
        if r.status_code != 200:
            logger.error(f"Error fetching article {doi}: {r.status_code}, {r.text}")
            return None
        logger.debug(f"Successfully retrieved article {doi}")
        return r.json()
    except Exception as e:
        logger.error(f"Error processing article {doi}: {e}")
        return None

def decode_unicode(text):
    """
    Normalize and decode Unicode characters to avoid encoding issues.
    
    Args:
        text (str): Unicode text to normalize
        
    Returns:
        str: Normalized text with combining characters removed
    """
    nfkd_form = unicodedata.normalize('NFKD', text)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def process_article_data(article_data):
    """
    Extract and process relevant information from Elsevier article data.
    
    Args:
        article_data (dict): Raw article data from Elsevier API
        
    Returns:
        dict: Processed article metadata including authors, date, DOI, title, abstract, etc.
    """
    # Extract core data from the response
    core_data = article_data.get("full-text-retrieval-response", {}).get("coredata", {})
    
    # Process authors information
    creators = core_data.get("dc:creator", [])
    authors = [decode_unicode(author['$']) for author in creators if isinstance(author, dict) and '$' in author]
    
    # Create a structured dictionary with all relevant fields
    processed_data = {
        "authors": authors,
        "date_of_publication": core_data.get("prism:coverDate", None),
        "doi": core_data.get("prism:doi", None),
        "text": article_data.get("originalText", ""),
        "title": core_data.get("dc:title", None),
        "abstract": core_data.get("dc:description", None),
        "publication_name": core_data.get("prism:publicationName", None),
        "category": "optometry"
    }
    return processed_data

def save_processed_article(processed_data, output_filename):
    """
    Save processed article data to a JSON file.
    
    Args:
        processed_data (dict): Processed article data to save
        output_filename (str): Filename for the output JSON file
        
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs('processed_articles', exist_ok=True)
    
    # Construct full output path
    output_path = os.path.join('processed_articles', output_filename)
    
    # Write the data to a JSON file
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(processed_data, output_file, indent=4)
    
    logger.info(f"Saved processed article to {output_path}")

def run_queries(queries: list[str], max_results: int = 10) -> list[ResearchPaper]:
    """
    Execute each query against the Elsevier service and return a flat list of ResearchPaper objects.
    
    This function:
    1. Loads API credentials from environment variables
    2. Executes each query against Elsevier's API
    3. Retrieves full article data for each matching DOI
    4. Processes the article data into ResearchPaper objects
    5. Stores the papers in the database
    
    Args:
        queries: List of search queries to execute against Elsevier
        max_results: Maximum number of results to return per query
        
    Returns:
        List of ResearchPaper objects from query results
    """
    import os
    from dotenv import load_dotenv
    
    # Load API credentials from environment variables
    load_dotenv()
    api_key = os.getenv('ELSEVIER_API_KEY')
    insttoken = os.getenv('ELSEVIER_INSTTOKEN')
    
    if not api_key or not insttoken:
        logger.error("Missing required environment variables: ELSEVIER_API_KEY or ELSEVIER_INSTTOKEN")
        return []
        
    papers = []
    
    for q in queries:
        # Look for DOIs that match the query
        dois = search_articles(query=q, api_key=api_key, insttoken=insttoken)
        
        # Limit to max_results
        dois = dois[:max_results]
        
        # For each DOI, fetch the full paper details
        for doi in dois:
            article_data = get_full_text_article(doi, api_key, insttoken)
            if article_data:
                try:
                    coredata = article_data.get('full-text-retrieval-response', {}).get('coredata', {})
                    title = coredata.get('dc:title')
                    abstract = coredata.get('dc:description')
                    
                    # Skip articles without title or abstract
                    if not title or not abstract:
                        logger.warning(f"Article missing title or abstract: {doi}")
                        continue
                    
                    # Process article data into the expected format
                    processed_data = process_article_data(article_data)
                    
                    # Create a ResearchPaper object
                    paper = ResearchPaper(
                        doi=processed_data.get("doi", ""),
                        title=processed_data.get("title", ""),
                        authors="; ".join(processed_data.get("authors", [])),
                        date=processed_data.get("date_of_publication", ""),
                        abstract=processed_data.get("abstract", ""),
                        category=processed_data.get("category", ""),
                        license="",  # Not available from Elsevier API
                        version="",  # Not applicable for published papers
                        author_corresponding="",  # Not directly available
                        author_corresponding_institution="",  # Not directly available
                        published_journal=processed_data.get("publication_name", ""),
                        published_date=processed_data.get("date_of_publication", ""),
                        published_doi=processed_data.get("doi", ""),
                        inclusion_decision=None,
                        criteria_assessment=None,
                        assessment_explanation=None,
                        assessment_datetime=None
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error processing article data for {doi}: {e}")
    
    # Upsert all papers to the database
    if papers:
        logger.info(f"Upserting {len(papers)} papers to database")
        upsert_elsevier_papers(papers)
    
    return papers

def upsert_elsevier_papers(papers: list[ResearchPaper]):
    """
    Upsert a list of ResearchPaper objects into the database using PostgreSQL's
    INSERT ... ON CONFLICT DO UPDATE statement.
    
    This function manages the database session and handles the upsert operation
    which either inserts new papers or updates existing ones based on the DOI.
    
    Args:
        papers: List of ResearchPaper objects to insert or update
        
    Returns:
        Tuple of (inserted_count, updated_count) for tracking purposes
    """
    if not papers:
        logger.info("No papers to upsert")
        return (0, 0)
    
    session = get_db_session()
    inserted = 0
    updated = 0
    
    try:
        # Get the table from the ResearchPaper model
        table = ResearchPaper.__table__
        
        for paper in papers:
            # Convert the ResearchPaper object to a dictionary
            paper_dict = paper.to_dict(exclude=["id"])
            
            # Create the insert statement
            stmt = insert(table).values(paper_dict)
            
            # Set up the ON CONFLICT DO UPDATE clause
            # specify which columns should be updated during an ON CONFLICT DO UPDATE operation in SQLAlchemy
            update_dict = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in ["doi", "id"]}
            stmt = stmt.on_conflict_do_update(
                index_elements=["doi"],
                set_=update_dict
            )
            
            # Execute the statement
            result = session.execute(stmt)
            
            # Check if the paper was inserted or updated
            if result.rowcount > 0:
                inserted += 1
            else:
                updated += 1
                
        # Commit the transaction
        session.commit()
        logger.info(f"Upserted {len(papers)} papers ({inserted} inserted, {updated} updated)")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error upserting papers: {e}")
    finally:
        close_db_session(session)
        
    return (inserted, updated)

def fetch_full_text(paper: ResearchPaper) -> str:
    """
    Fetches the full text or enhanced metadata for an Elsevier paper.
    
    Args:
        paper (ResearchPaper): A ResearchPaper object with a DOI
        
    Returns:
        str: The full text or enhanced metadata of the paper if available,
             or the abstract if full text cannot be retrieved
    """
    logger.info(f"Fetching full text for Elsevier paper with DOI: {paper.doi}")
    
    # Load environment variables for API credentials
    load_dotenv()
    api_key = os.getenv('ELSEVIER_API_KEY')
    insttoken = os.getenv('ELSEVIER_INSTTOKEN')
    
    if not api_key or not insttoken:
        logger.error("Missing required environment variables: ELSEVIER_API_KEY or ELSEVIER_INSTTOKEN")
        return paper.abstract if paper.abstract else ""
    
    try:
        # If no DOI is available, return the abstract as fallback
        if not paper.doi:
            logger.warning("No DOI provided for Elsevier paper")
            return paper.abstract if paper.abstract else ""
        
        # Fetch full text article data
        article_data = get_full_text_article(paper.doi, api_key, insttoken)
        if not article_data:
            logger.warning(f"Could not fetch full text data for DOI: {paper.doi}")
            return paper.abstract if paper.abstract else ""
        
        # Process the article data
        processed_data = process_article_data(article_data)
        
        # Compile all available information into a comprehensive text
        components = []
        
        # Add title
        if processed_data.get("title"):
            components.append(f"Title: {processed_data.get('title')}")
        
        # Add abstract
        if processed_data.get("abstract"):
            components.append(f"Abstract: {processed_data.get('abstract')}")
        
        # Add authors
        if processed_data.get("authors"):
            components.append(f"Authors: {', '.join(processed_data.get('authors'))}")
        
        # Add publication date
        if processed_data.get("date_of_publication"):
            components.append(f"Publication Date: {processed_data.get('date_of_publication')}")
        
        # Add publication name
        if processed_data.get("publication_name"):
            components.append(f"Journal: {processed_data.get('publication_name')}")
        
        # Add DOI reference
        components.append(f"DOI: {paper.doi}")
        
        # Add main text if available
        if processed_data.get("text") and len(processed_data.get("text")) > 0:
            # Truncate text if it's too long (often Elsevier has full text which can be very lengthy)
            text = processed_data.get("text")
            max_length = 10000  # Set a reasonable maximum length
            if len(text) > max_length:
                text = text[:max_length] + "... [truncated]"
            components.append(f"Text: {text}")
        
        # Combine all components
        enhanced_text = "\n\n".join(components)
        logger.info(f"Successfully compiled enhanced information for Elsevier DOI {paper.doi}")
        return enhanced_text
        
    except Exception as e:
        logger.error(f"Error fetching full text for Elsevier paper: {e}")
        return paper.abstract if paper.abstract else ""

def main():
    """
    Main function for running the Elsevier service as a standalone script.
    
    This function:
    1. Loads environment variables
    2. Searches for articles using a predefined query
    3. Fetches full text for each matching article
    4. Processes and saves the article data to JSON files
    
    Returns:
        None
    """
    # Load environment variables for API credentials
    load_dotenv()
    api_key = os.getenv('ELSEVIER_API_KEY')
    insttoken = os.getenv('ELSEVIER_INSTTOKEN')
    search_query = input("Enter a keyword to search for articles: ").strip()
    if not search_query:
        logger.error("No keyword entered. Exiting.")
        return
    
    if not api_key or not insttoken:
        logger.error("Missing required environment variables: ELSEVIER_API_KEY or ELSEVIER_INSTTOKEN")
        return
        
    logger.info(f"Starting Elsevier search with query: {search_query}")
    dois = search_articles(search_query, api_key, insttoken)
    
    if not dois:
        logger.warning("No articles found or error occurred during search")
        return

    for doi in dois:
        logger.info(f"Fetching full text for DOI: {doi}")
        article_data = get_full_text_article(doi, api_key, insttoken)

        if article_data:
            try:
                coredata = article_data.get('full-text-retrieval-response', {}).get('coredata', {})
                title = coredata.get('dc:title')
                abstract = coredata.get('dc:description')
                if not title:
                    logger.warning("Article data retrieved but no title found. Skipping.")
                    continue
                if not abstract:
                    logger.warning("Article has no abstract. Skipping.")
                    continue
                logger.info(f"Successfully retrieved article: {title}")
                processed_data = process_article_data(article_data)
                output_filename = f'{doi.replace("/", "_")}.json'
                save_processed_article(processed_data, output_filename)
                logger.info(f"Saved article data to {output_filename}")
                logger.debug("-" * 50)
            except Exception as e:
                logger.error(f"Error processing article data: {e}")
        else:
            logger.warning(f"Failed to retrieve article data for {doi}")
        
    saved_articles = len([
        name for name in os.listdir('processed_articles')
        if os.path.isfile(os.path.join('processed_articles', name)) and name.endswith('.json')
    ])
    logger.info(f"Total processed articles saved: {saved_articles}")


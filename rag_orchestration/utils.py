# rag_orchestration/utils.py

import difflib
import importlib
import json
import logging
import re
import os
from functools import lru_cache
from jinja2 import Template
from pathlib import Path
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.orm import Session
from types import ModuleType
from typing import List, Dict, Any, Set, Union

from data.db import get_db_session, close_db_session
from data.ingest import store_chunks_for_paper, prepare_and_embed_text, get_embedding_model
from data.models import PaperCitation, ResearchPaper, ResearchPaperChunk
from rag_orchestration.models import LLMQueryResponse
from prompts_library.rag import service_query_creation_template
from services.opencitations import load_citing_dois
from services.service_model import ServiceQueryInfo

from services.arxiv import fetch_full_text as get_arxiv_pdf_text
from services.pubmed import fetch_full_text as get_pubmed_pdf_text
from services.elsevier import fetch_full_text as get_elsevier_pdf_text
from services.unpaywall import fetch_full_text as get_unpaywall_pdf_text
from services.biorxiv import fetch_full_text as get_biorxiv_pdf_text


logger = logging.getLogger(__name__)

SERVICE_PRIORITY = [
    "elsevier",
    "pubmed",
    "unpaywall",
    "biorxiv",
    "arxiv"
]

FETCHERS = {
  "arxiv":   get_arxiv_pdf_text,
  "biorxiv": get_biorxiv_pdf_text,
  "elsevier": get_elsevier_pdf_text,
  "pubmed": get_pubmed_pdf_text,
  "unpaywall": get_unpaywall_pdf_text
}

def mask_abstract_differences(abstract_a: str, abstract_b: str, flag: str = "[[DIFF]]") -> str:
    """
    Compare two abstracts and replace differing spans with a flag.
    Returns a masked version of abstract_a (could use b just as well).
    """
    logger.debug(f"Masking differences between abstracts of lengths {len(abstract_a)} and {len(abstract_b)}")
    matcher = difflib.SequenceMatcher(None, abstract_a, abstract_b)
    masked = []
    last_end = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            masked.append(abstract_a[i1:i2])
        else:
            masked.append(flag)
    
    result = "".join(masked)
    diff_count = result.count(flag)
    logger.debug(f"Masking complete with {diff_count} difference markers")
    return result

def replace_bracketed_content(text):
    return re.sub(r'\[\[(.*?)\]\]', '[[DIFF]]', text)

@lru_cache(maxsize=32)
def load_queriable_services(directory="services") -> list[ServiceQueryInfo]:
    """
    Dynamically imports all .py files in the specified services directory and collects
    valid ServiceQueryInfo objects named `service_info`.

    Args:
        directory (str): Path to the directory containing service modules.

    Returns:
        List[ServiceQueryInfo]: A list of enabled and valid service info objects.
    """

    logger.info(f"Loading queryable services from directory: {directory}")
    service_infos = []

    for path in Path(directory).glob("*.py"):
        module_name = f"{directory.replace('/', '.')}.{path.stem}"
        try:
            logger.debug(f"Attempting to import module: {module_name}")
            module: ModuleType = importlib.import_module(module_name)
            if hasattr(module, "service_info"):
                service = module.service_info
                if isinstance(service, ServiceQueryInfo) and service.enabled:
                    logger.debug(f"Found enabled service: {service.name}")
                    service_infos.append(service)
                else:
                    logger.debug(f"Service {getattr(service, 'name', 'unknown')} exists but is disabled or invalid")
        except Exception as e:
            logger.error(f"Error loading {module_name}: {e}")

    logger.info(f"Loaded {len(service_infos)} enabled services")
    return service_infos

def build_multi_service_prompt(masked_abstract: str, services: List[ServiceQueryInfo], template_str: str) -> str:
    """
    Builds a multi-service query generation prompt using a Jinja2 template.

    Args:
        masked_abstract (str): The abstract with masked results for query generation.
        services (List[ServiceQueryInfo]): List of enabled service metadata.
        template_str (str): A Jinja2 template string.

    Returns:
        str: A formatted prompt to send to the LLM.
    """
    logger.debug(f"Building multi-service prompt with {len(services)} available services")
    template = Template(template_str)
    result = template.render(masked_abstract=masked_abstract, services=services)
    logger.debug(f"Generated prompt of length {len(result)} characters")
    return result

def generate_service_queries(
    masked_abstract: str, 
    llm_client: Any, 
    template_str: str,
    max_retries: int = 3
) -> Dict[str, List[str]]:
    """
    Generate search queries for each relevant service using an LLM.

    Args:
        masked_abstract (str): The abstract with masked results for query generation.
        llm_client: A client from services.llm_services for making LLM API calls.
        template_str (str): A Jinja2 template string for the query creation prompt.
        max_retries (int): Maximum number of retries for parsing LLM response.

    Returns:
        Dict[str, List[str]]: A dictionary mapping service names to lists of query strings.
    """
    # Load all available services
    logger.info("Generating search queries from masked abstract")
    services = load_queriable_services()
    
    # Build the prompt to send to the LLM
    prompt = build_multi_service_prompt(masked_abstract, services, template_str)
    
    # Generate queries using the LLM client
    query_plan = {}
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Call the LLM complete method to generate queries
            logger.debug(f"Calling LLM to generate queries (attempt {retry_count + 1}/{max_retries})")
            response = llm_client.complete(prompt)
            
            # Extract the JSON portion from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                logger.debug(f"Found JSON in response from index {json_start} to {json_end}")
                parsed_response = json.loads(json_str)
                
                # Convert to our model format
                llm_query_response = LLMQueryResponse(**parsed_response)
                
                # Create a service name to queries dictionary
                for service_query in llm_query_response.queries:
                    query_plan[service_query.service] = service_query.queries
                    logger.debug(f"Generated {len(service_query.queries)} queries for service '{service_query.service}'")
                
                logger.info(f"Successfully generated queries for {len(query_plan)} services")
                break  # Successfully parsed, exit the loop
            else:
                raise ValueError("No JSON found in LLM response")
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing LLM response (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"Failed to generate valid queries after {max_retries} attempts")
                return {}
    
    return query_plan

def dispatch_queries(
    query_plan: Dict[str, List[str]],
    max_results_per_query: int = 10
) -> Dict[str, List]:
    """
    Given a plan of { service_name: [queries...] }, call each service's run_queries
    and return their results.

    Returns:
        Dict mapping service_name → List of ResearchPaper (or raw dicts)
    """
    logger.info(f"Dispatching queries to {len(query_plan)} services")
    services: List[ServiceQueryInfo] = load_queriable_services()
    name_to_info = {s.name: s for s in services}
    results = {}

    for service_name, queries in query_plan.items():
        if service_name not in name_to_info:
            logger.warning(f"Service '{service_name}' not found in available services, skipping")
            continue

        logger.info(f"Calling run_queries on service '{service_name}' with {len(queries)} queries")
        module_name = f"services.{service_name.lower().replace(' ', '')}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "run_queries"):
                papers = module.run_queries(queries, max_results=max_results_per_query)
                results[service_name] = papers
                logger.info(f"Service '{service_name}' returned {len(papers)} results")
            else:
                logger.error(f"{service_name} module has no run_queries() method")
                raise AttributeError(f"{service_name} module has no run_queries()")
        except ImportError as e:
            logger.error(f"Could not import module {module_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing queries for '{service_name}': {e}")
            raise

    total_results = sum(len(results_list) for results_list in results.values())
    logger.info(f"Query dispatch complete, retrieved {total_results} total results across {len(results)} services")
    return results

@lru_cache(maxsize=128)
def load_citing_dois_from_db(test_doi: str) -> Set[str]:
    """
    Load all DOIs of papers that cite `test_doi` from the local paper_citations table.
    Results are cached using lru_cache to improve performance.

    Args:
        test_doi: The DOI of the test paper (maps to PaperCitation.doi).

    Returns:
        A set of DOIs (PaperCitation.cited_paper_doi) for papers that cite the test paper.
    """
    logger.info(f"Loading citing DOIs for test paper with DOI: {test_doi}")
    session = get_db_session()
    try:
        rows = (
            session
            .query(PaperCitation.cited_paper_doi)
            .filter(PaperCitation.doi == test_doi)
            .all()
        )
        # rows is a list of one‑tuples, e.g. [("10.1234/xyz",), …]
        result = {r[0] for r in rows if r[0]}
        logger.info(f"Found {len(result)} citing papers in database for DOI: {test_doi}")
        return result
    except Exception as e:
        logger.error(f"Error loading citing DOIs from database: {e}")
        raise
    finally:
        close_db_session(session)

def cull_citing_papers(
    papers: List[ResearchPaper],
    test_doi: str,
    include_opencitations: bool = True
) -> List[ResearchPaper]:
    """
    Remove any papers from `papers` whose DOI appears in either:
      1. The local paper_citations table (cited_paper_doi entries for test_doi), or
      2. (Optionally) the OpenCitations API results.
    
    Uses cached results from previous calls with the same test_doi.

    Args:
        papers: List of ResearchPaper objects returned by your RAG services.
        test_doi: DOI of the test abstract paper.
        include_opencitations: If True, also exclude DOIs from the OpenCitations lookup.

    Returns:
        Filtered list of ResearchPaper.
    """
    # Get excluded DOIs from local database (function is cached with lru_cache)
    excluded = load_citing_dois_from_db(test_doi)
    logger.debug(f"Loaded {len(excluded)} citing DOIs from database cache")

    # Add external citations if requested (function is cached with lru_cache)
    if include_opencitations:
        try:
            opencitations_dois = load_citing_dois(test_doi)
            logger.debug(f"Loaded {len(opencitations_dois)} citing DOIs from OpenCitations cache")
            excluded |= opencitations_dois
        except Exception as e:
            # Log & swallow; we still filter by local citations
            logger.warning(f"Failed to get citing DOIs from OpenCitations: {e}")
    
    total_excluded = len(excluded)
    logger.info(f"Total of {total_excluded} citing DOIs to exclude")

    # Filter out any papers whose DOI is in excluded set
    # Normalize DOIs (lowercase) for case-insensitive comparison
    excluded_normalized = {d.lower() for d in excluded}

    def is_allowed(p: ResearchPaper) -> bool:
        doi = (p.doi or "").lower()
        return doi not in excluded_normalized

    filtered_papers = [p for p in papers if is_allowed(p)]
    logger.info(f"Filtered {len(papers) - len(filtered_papers)} citing papers from a total of {len(papers)}")
    
    return filtered_papers

@lru_cache(maxsize=256)
def resolve_svc(doi: str, return_list: bool = False) -> Union[str, List[str]]:
    """
    Determine the appropriate service(s) to use based on a paper's DOI.
    Uses explicit DOI pattern matching first, then falls back to SERVICE_PRIORITY order.
    
    Args:
        doi (str): The Digital Object Identifier of the paper
        return_list (bool): If True, returns a list of services in priority order;
                           If False, returns only the highest priority service
        
    Returns:
        Union[str, List[str]]: Either the highest priority service name or a list of service names
    """
    if not doi:
        logger.warning("Empty DOI passed to resolve_svc")
        raise ValueError("Cannot resolve service for empty DOI")
        
    doi = doi.lower()
    
    # Build a prioritized list of services based on the DOI
    services = []
    
    # First check for known DOI patterns
    # Check for arXiv identifiers
    if doi.startswith("arxiv:") or "arxiv" in doi:
        services.append("arxiv")
        
    # Check for bioRxiv/medRxiv DOIs
    if doi.startswith("10.1101/"):
        services.append("biorxiv")
        
    # Check for Elsevier DOIs - these typically start with 10.1016
    if doi.startswith("10.1016/"):
        services.append("elsevier")
    
    # For PubMed papers - many life science journals
    if doi.startswith("10.1093/") or doi.startswith("10.1097/") or doi.startswith("10.1001/"):
        services.append("pubmed")
    
    # If we haven't added any services yet based on specific patterns,
    # or to fill out the remaining services in the priority list
    for service in SERVICE_PRIORITY:
        if service not in services and service in FETCHERS:
            services.append(service)
    
    # Filter out services that don't have fetchers
    services = [svc for svc in services if svc in FETCHERS]
    
    if not services:
        logger.warning(f"No service match for DOI {doi}")
        if return_list:
            return []
        return "pubmed"  # Default fallback
    
    if return_list:
        return services
    else:
        # Return just the highest priority service
        return services[0]

@lru_cache(maxsize=64)
def get_chunks_from_db(session: Session, doi: str) -> List[Dict]:
    chunks = (
        session.query(ResearchPaperChunk)
        .filter(ResearchPaperChunk.doi == doi)
        .order_by(ResearchPaperChunk.chunk_index)
        .all()
    )
    return [
        {
            "chunk_index": chunk.chunk_index,
            "chunk_text": chunk.chunk_text,
            "embedding": chunk.embedding,
            "doi": chunk.doi,
            "section": chunk.section,
        }
        for chunk in chunks
    ]

def get_or_fetch_research_paper_text(paper: ResearchPaper, session=None):
    """
    Get text for a research paper, either from the database or by fetching from appropriate service.
    
    Args:
        paper (ResearchPaper): The paper object with at least a DOI
        session: Optional SQLAlchemy session. If None, a new session will be created.
        
    Returns:
        str: The full text of the paper
    """
    logger.info(f"Getting text for paper with DOI: {paper.doi}")
    
    # If session wasn't provided, create a new one
    close_after = False
    if session is None:
        session = get_db_session()
        close_after = True
        
    try:
        # 0) Ensure paper exists in database before proceeding
        db_paper = session.query(ResearchPaper).filter(ResearchPaper.doi == paper.doi).first()
        if not db_paper:
            logger.info(f"Paper with DOI {paper.doi} not found in database, adding it first")
            session.add(paper)
            session.commit()
            # Now paper.id should be populated
        elif paper.id is None:
            # If our paper object doesn't have an ID but exists in DB, use the DB version's ID
            paper.id = db_paper.id
            
        # 1) Look in research_paper_chunks table
        chunks = get_chunks_from_db(session, paper.doi)
        
        if chunks:
            logger.info(f"Found {len(chunks)} existing chunks in database for DOI: {paper.doi}")
            # Combine all chunks in order of chunk_index
            full_text = "\n\n".join([chunk["chunk_text"] for chunk in chunks])
            return full_text
            
        # 2) Otherwise attempt to fetch from services
        logger.info(f"No chunks found in database. Fetching from service for DOI: {paper.doi}")
        services = resolve_svc(paper.doi, return_list=True)
        
        for service_name in services:
            if service_name not in FETCHERS:
                logger.warning(f"No fetcher found for service {service_name}")
                continue
            
            fetcher = FETCHERS[service_name]
            logger.info(f"Using {service_name} service to fetch text for DOI: {paper.doi}")
            
            try:
                # Use the appropriate service to fetch text
                full_text = fetcher(paper)
                
                if full_text:
                    logger.info(f"Successfully fetched text from {service_name} for DOI: {paper.doi}")
                    
                    # 3) If we fetched from service, store the text in the db
                    try:
                        logger.info(f"Processing and storing text for DOI: {paper.doi}")
                        
                        # Get the embedding model
                        embedder = get_embedding_model()
                        if not embedder:
                            logger.error("No embedding model available for text processing")
                            return full_text
                        
                        # Prepare and embed the text
                        embedded_chunks = prepare_and_embed_text(full_text, paper.doi, embedder)
                        
                        # Store the chunks in the database
                        if embedded_chunks:
                            store_chunks_for_paper(session, paper.id, paper.doi, embedded_chunks)
                            logger.info(f"Stored {len(embedded_chunks)} chunks for DOI: {paper.doi}")
                            
                    except Exception as e:
                        logger.error(f"Error storing text chunks for DOI {paper.doi}: {e}")
                        # If we failed to store chunks, we still return the text we fetched
                    
                    # 4) Return the text
                    return full_text
                
            except Exception as e:
                logger.warning(f"Failed to fetch text from {service_name} for DOI: {paper.doi}: {e}")
                continue
        
        logger.warning(f"All services failed to fetch text for DOI: {paper.doi}")
        return paper.abstract if paper.abstract else ""
        
    except Exception as e:
        logger.error(f"Error in get_or_fetch_research_paper_text for DOI {paper.doi}: {e}")
        if session:
            session.rollback()  # Explicitly roll back the session
        # Fallback to abstract
        return paper.abstract if paper.abstract else ""
        
    finally:
        # Close session if we created it
        if close_after:
            close_db_session(session)

def store_chunks_for_paper(session: Session, paper_id: int, doi: str, embedded_chunks: List[dict]) -> int:
    """
    Store the embedded chunks for a given paper in the database.
    
    Args:
        session (Session): SQLAlchemy database session
        paper_id (int): The ID of the paper
        doi (str): The DOI of the paper
        embedded_chunks (List[dict]): List of embedded chunks to store
        
    Returns:
        int: The number of chunks stored
    """
    logger.info(f"Storing {len(embedded_chunks)} chunks for paper ID {paper_id} and DOI {doi}")
    for chunk in embedded_chunks:  # FIX: use embedded_chunks instead of chunks
        chunk_obj = ResearchPaperChunk(
            paper_id=paper_id,
            doi=doi,
            chunk_index=chunk['chunk_index'],
            chunk_text=chunk['chunk_text'],
            embedding=chunk['embedding'],
            section=chunk.get('section')
        )
        session.add(chunk_obj)
    session.commit()
    return len(embedded_chunks)

def generate_rag_context(session: Session, query_text: str, k: int = 8, return_chunk_count: bool = False):
    """
    Generate RAG context by retrieving the most semantically similar chunks to the query.
    
    Args:
        session (Session): SQLAlchemy database session
        query_text (str): The query text to find relevant document chunks for
        k (int, optional): The number of top similar chunks to retrieve. Defaults to 8.
        return_chunk_count (bool, optional): If True, return a tuple of (rag_text, chunks_count).
        
    Returns:
        Union[str, Tuple[str, int]]: A concatenated string of text chunks to be used as context for RAG,
                                    or a tuple with the string and count of chunks if return_chunk_count=True
    """
    embedder = get_embedding_model()
    embedding = embedder.embed_query(query_text)

    top_chunks = fetch_top_k_similar_chunks(session, embedding, k)
    rag_text = "\n\n".join(chunk.chunk_text for chunk in top_chunks)
    
    return (rag_text, len(top_chunks)) if return_chunk_count else (rag_text, None)

def fetch_top_k_similar_chunks(session: Session, embedding: list[float], k: int = 8):
    """
    Retrieve the top k chunks most similar to the given embedding vector.
    
    Args:
        session (Session): SQLAlchemy database session
        embedding (list[float]): The embedding vector to compare against stored chunks
        k (int, optional): The number of similar chunks to retrieve. Defaults to 8.
        
    Returns:
        list[ResearchPaperChunk]: A list of the k most similar document chunks
    """
    order_expr = ResearchPaperChunk.embedding.cosine_distance(embedding)
    return (
        session.scalars(
            select(ResearchPaperChunk).order_by(order_expr).limit(k)
        ).all()
    )
def assemble_test_abstract_prompt(rag_text: str, abstract_text: str) -> str:
    """
    Assemble the prompt for the test abstract using Jinja2 templating.
    
    Args:
        rag_text (str): The RAG context text
        abstract_text (str): The test abstract text
        
    Returns:
        str: The formatted prompt text
    """
    from prompts_library.rag import rag_context_template
    template = Template(rag_context_template)
    # Properly include both the RAG context and abstract text in the template
    return template.render(rag_context=rag_text, abstract=abstract_text)



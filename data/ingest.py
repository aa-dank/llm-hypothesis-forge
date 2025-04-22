#data/ingest.py

import os
import logging
import nltk
import time
import re
import datetime
import pyalex
import pandas as pd
import xml.etree.ElementTree as ET
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import List
from data.models import ResearchPaper, ResearchPaperChunk
from data.db import init_db, get_db_session, close_db_session

from services.arxiv import (
    arxiv_service, 
    extract_arxiv_id_from_doi, 
    fetch_arxiv_paper, 
    fetch_arxiv_details, 
    arxiv_taxonomy
)

from services.biorxiv import (
    create_biorxiv_paper, 
    fetch_biorxiv_details, 
    fetch_biorxiv_publication_info, 
    process_publication_info, 
    import_biorxiv_paper, 
    fetch_biorxiv_paper_by_doi
)

logger = logging.getLogger(__name__)  # Module-level logger

EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns an instance of the embedding model for text processing.
    
    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings model.
    """
    # You can replace this with any other embedding model you prefer
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)

def get_tokenizer(chunk_size: int = 500, chunk_overlap: int = 50, model_name: str = "text-embedding-3-small"):
    """Returns a RecursiveCharacterTextSplitter for token-based splitting."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name
    )

def add_paper_to_db(paper, session):
    """Add a paper to the database with error handling."""
    try:
        session.add(paper)
        session.flush()
        return True
    except IntegrityError:
        session.rollback()
        logger.info(f"Paper with DOI {paper.doi} already exists in database (detected by constraint)")
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding paper to database: {e}")
        return False

def batch_commit(session, count=10):
    """Perform a database commit after a certain number of operations."""
    if count % 10 == 0:
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.error("Unexpected integrity error during batch commit")
        except Exception as e:
            session.rollback()
            logger.error(f"Error during batch commit: {e}")

def final_commit(session):
    """Perform the final database commit with error handling."""
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        logger.error("Unexpected integrity error during final commit")
    except Exception as e:
        session.rollback()
        logger.error(f"Error during final commit: {e}")

def download_preprints_with_publication_info(start_date, end_date, n_cursor, output_format, source="biorxiv", 
                                            category=None, published_in_journal=None):
    """
    Downloads preprint metadata from biorxiv/arxiv and stores it directly in database using SQLAlchemy.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        n_cursor (int): Number of pagination cursors to iterate through
        output_format (str): Output format (json for biorxiv)
        source (str): Data source - "biorxiv" or "arxiv"
        category (str): Optional - Category/domain to filter by (e.g., "q-bio.NC", "neuroscience")
        published_in_journal (str): Optional - Filter for papers published in a specific journal
    """
    # Initialize database and get a session
    init_db()
    session = get_db_session()
    
    try:
        if source.lower() == "biorxiv":
            return download_biorxiv_preprints(start_date, end_date, n_cursor, output_format, session, 
                                             category, published_in_journal)
        elif source.lower() == "arxiv":
            return download_arxiv_preprints(start_date, end_date, n_cursor, session, 
                                           category, published_in_journal)
        else:
            raise ValueError("Unsupported source. Use 'biorxiv' or 'arxiv'")
    finally:
        close_db_session(session)

def download_biorxiv_preprints(start_date, end_date, n_cursor, output_format, session, category=None, published_in_journal=None):
    """
    Downloads preprint metadata from biorxiv and stores it directly in database using SQLAlchemy.
    """
    interval = f'{start_date}/{end_date}'
    
    record_count = 0
    duplicate_count = 0
    filtered_count = 0
    
    # Keep track of processed DOIs to avoid duplicates within the same run
    processed_dois = set()
    
    for cursor in range(0, n_cursor):
        response = fetch_biorxiv_details(interval, cursor, output_format)
        if not response:
            continue
            
        data = response.json()
        records = data.get('collection', [])
        logger.debug(f"Found {len(records)} records on page {cursor}.")
        
        # Check if we've reached the end of available results
        if len(records) == 0:
            logger.info(f"No more records available after page {cursor}.")
            break
        
        new_records_on_page = 0
        
        for record in records:
            # Get DOI for lookups
            doi = record.get("doi", "")
            
            # Skip records with missing DOI
            if not doi:
                logger.debug("Skipping record with missing DOI")
                continue
                
            # Skip if we've already processed this DOI in this run
            if doi in processed_dois:
                logger.debug(f"Already processed DOI {doi} in this run - skipping")
                continue
                
            # Add to our tracking set
            processed_dois.add(doi)
            
            # Check if paper already exists in the database
            existing_paper = session.query(ResearchPaper).filter(ResearchPaper.doi == doi).first()
            if existing_paper:
                logger.debug(f"Paper with DOI {doi} already exists in database (pre-check)")
                duplicate_count += 1
                continue
            
            # Create a new paper object
            paper = create_biorxiv_paper(record)
            
            # Fetch additional publication info if available
            response = fetch_biorxiv_publication_info(doi, output_format)
            if response:
                process_publication_info(response, paper)
            
            # Filter by category if specified
            if category and paper.category:
                if category.lower() not in paper.category.lower():
                    filtered_count += 1
                    continue
                    
            # Filter by published journal if specified
            if published_in_journal and paper.published_journal:
                if published_in_journal.lower() not in paper.published_journal.lower():
                    filtered_count += 1
                    continue
            
            # Add the paper to the database
            if add_paper_to_db(paper, session):
                record_count += 1
                new_records_on_page += 1
                batch_commit(session, record_count)
            else:
                duplicate_count += 1
                
            # Sleep to avoid API rate limits
            time.sleep(0.1)
        
        logger.info(f"Page {cursor}: Added {new_records_on_page} new records")
        
        # If no new records were added on this page, we might have reached the end
        if new_records_on_page == 0 and cursor > 0:
            logger.info(f"No new records found on page {cursor}; ending data retrieval.")
    
    # Final commit
    final_commit(session)
    
    logger.info(f"Saved {record_count} new records; skipped {duplicate_count} duplicates.")
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} records not matching criteria.")
    
    return record_count, duplicate_count

def download_arxiv_preprints(start_date, end_date, n_cursor, session: Session, category=None, published_in_journal=None):
    """
    Downloads preprint metadata from arXiv and stores it directly in the database using SQLAlchemy.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        n_cursor (int): Number of pagination cursors to iterate through
        session (Session): SQLAlchemy database session
        category (str): Optional - Category/domain to filter by (e.g., "cs.AI", "q-bio.NC")
        published_in_journal (str): Optional - Filter for papers published in a specific journal
    """
    # Convert dates to arXiv format (YYYYMMDD)
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    start_date_arxiv = start_dt.strftime("%Y%m%d")
    end_date_arxiv = end_dt.strftime("%Y%m%d")
    
    if category:
        if " OR " in category:
            # For multiple categories already formatted with cat: prefixes
            search_query = f'({category}) AND submittedDate:[{start_date_arxiv} TO {end_date_arxiv}]'
        else:
            # For a single category
            search_query = f'cat:{category} AND submittedDate:[{start_date_arxiv} TO {end_date_arxiv}]'
    else:
        search_query = f'submittedDate:[{start_date_arxiv} TO {end_date_arxiv}]'

    logger.info(f"Using arXiv search query: {search_query}")
    
    max_results = 100  # arXiv API max limit per request
    
    record_count = 0
    duplicate_count = 0
    filtered_count = 0
    
    # Keep track of processed arXiv IDs to avoid duplicates within the same run
    processed_arxiv_ids = set()
    
    # arXiv namespaces for parsing XML
    namespaces = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    # Make requests for each batch
    for i in range(n_cursor):
        start_idx = i * max_results
        logger.info(f"Fetching arXiv data: batch {i+1}/{n_cursor}, start_idx={start_idx}")
        
        response = fetch_arxiv_details(search_query, start_idx, max_results)
        if not response:
            continue
            
        # Parse the XML response
        try:
            root = ET.fromstring(response.content)
            entries = root.findall('.//atom:entry', namespaces)
            logger.debug(f"Found {len(entries)} records in this batch.")
            
            new_records_in_batch = 0
            
            for entry in entries:
                # Extract arXiv ID for tracking
                arxiv_id = entry.find('./atom:id', namespaces).text.split('/')[-1]
                
                # Skip if we've already processed this ID in this run
                if arxiv_id in processed_arxiv_ids:
                    logger.debug(f"Already processed arXiv ID {arxiv_id} in this run - skipping")
                    continue
                    
                # Add to our tracking set
                processed_arxiv_ids.add(arxiv_id)
                
                # Create a paper object
                paper = arxiv_service.create_arxiv_paper(entry, namespaces)
                
                # Pre-check for existing papers in the database
                existing_paper = session.query(ResearchPaper).filter(ResearchPaper.doi == paper.doi).first()
                if existing_paper:
                    logger.debug(f"Paper with DOI {paper.doi} already exists in database (pre-check)")
                    duplicate_count += 1
                    continue
                
                # Apply journal filter if specified
                if published_in_journal and paper.published_journal:
                    if published_in_journal.lower() not in paper.published_journal.lower():
                        filtered_count += 1
                        continue
                
                # Add the paper to the database
                if add_paper_to_db(paper, session):
                    record_count += 1
                    new_records_in_batch += 1
                    batch_commit(session, record_count)
                else:
                    duplicate_count += 1
            
            logger.info(f"Added {new_records_in_batch} new records from batch {i+1}")
            
            # Break early if we found no new records in this batch
            if new_records_in_batch == 0 and i > 0:
                logger.info(f"No new records found in batch {i+1}, possibly reached end of available data.")
                break
                
            # Sleep to avoid API rate limits
            time.sleep(3)  # arXiv encourages responsible use with delays
            
        except Exception as e:
            logger.error(f"Error processing arXiv data: {e}")
    
    # Final commit
    final_commit(session)
    
    logger.info(f"Saved {record_count} new arXiv records to database.")
    logger.info(f"Skipped {duplicate_count} duplicate records.")
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} records not matching criteria.")
    
    return record_count, duplicate_count

def normalize_doi(doi):
    """Normalize a DOI by removing prefixes and standardizing format."""
    if not doi or not isinstance(doi, str):
        return None
        
    # Remove 'https://doi.org/' prefix if present
    if doi.lower().startswith('https://doi.org/'):
        return doi[16:].strip()
    return doi.strip()

def import_brainbench_papers(brainbench_csv_paths, output_format='json'):
    """
    Imports papers from BrainBench dataset CSV files into the database.
    """
    # Initialize database and get a session
    init_db()
    session = get_db_session()
    
    papers_added = 0
    papers_skipped = 0
    processed_dois = set()
    
    try:
        # Collect all DOIs from the BrainBench CSV files
        all_dois = []
        
        for csv_path in brainbench_csv_paths:
            if not os.path.exists(csv_path):
                logger.warning(f"CSV file {csv_path} not found.")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                
                if 'doi' not in df.columns:
                    logger.warning(f"'doi' column not found in {csv_path}")
                    continue
                    
                # Extract normalized DOIs
                for doi in df['doi']:
                    normalized_doi = normalize_doi(doi)
                    if normalized_doi:
                        all_dois.append(normalized_doi)
            
            except Exception as e:
                logger.error(f"Error processing {csv_path}: {e}")
        
        logger.info(f"Found {len(all_dois)} DOIs in BrainBench CSV files")
        
        # Process each unique DOI
        for doi in set(all_dois):
            if doi in processed_dois:
                continue
                
            processed_dois.add(doi)
            
            # Check if the paper already exists in the database
            existing_paper = session.query(ResearchPaper).filter(
                (ResearchPaper.doi == doi) | 
                (ResearchPaper.published_doi == doi)
            ).first()
            
            if existing_paper:
                logger.info(f"Paper with DOI {doi} already exists in database, marking as previously in BrainBench")
                existing_paper.previously_in_brainbench = True
                papers_skipped += 1
                continue
            
            # Determine if this is a biorXiv or arXiv paper based on DOI format
            if doi.startswith('10.1101/'):
                # This is a biorXiv paper
                logger.info(f"Fetching biorXiv metadata for DOI: {doi}")
                paper = import_biorxiv_paper(doi, session)
                
            elif 'arxiv' in doi.lower():
                # This appears to be an arXiv paper
                logger.info(f"Fetching arXiv metadata for DOI: {doi}")
                arxiv_id = doi.split(':')[-1] if ':' in doi else doi
                paper = arxiv_service.import_arxiv_paper(arxiv_id, session)
                
            else:
                # This is likely a published journal paper
                logger.info(f"Creating minimal entry for published paper DOI: {doi}")
                # Creating a minimal entry for a published paper
                paper = ResearchPaper(
                    doi=doi,
                    published_doi=doi,
                    previously_in_brainbench=True
                )
            
            # If we successfully created a paper object, add it to the database
            if paper:
                if add_paper_to_db(paper, session):
                    papers_added += 1
                    logger.info(f"Added paper with DOI {doi} to database")
                else:
                    papers_skipped += 1
                
                # Commit every 10 successfully added papers
                batch_commit(session, papers_added)
            
            # Sleep a bit to avoid overwhelming the APIs
            time.sleep(0.5)
        
        # Final commit
        final_commit(session)
        
    except Exception as e:
        logger.error(f"Error importing BrainBench papers: {e}")
    finally:
        close_db_session(session)
    
    logger.info(f"Added {papers_added} papers from BrainBench dataset")
    logger.info(f"Skipped {papers_skipped} papers (already in database)")
    
    return papers_added, papers_skipped

def fetch_paper_metadata_by_doi(doi):
    """
    Fetch paper metadata from arXiv based on a DOI.
    
    Args:
        doi (str): A DOI or identifier for the paper
    
    Returns:
        ResearchPaper or None: A ResearchPaper object with fetched metadata, or None if retrieval fails
    """
    logger.info(f"Attempting to fetch metadata for DOI: {doi}")
    
    # Handle arXiv DOIs
    arxiv_id = extract_arxiv_id_from_doi(doi)
    if arxiv_id:
        logger.info(f"Extracted arXiv ID: {arxiv_id}")
        return arxiv_service.fetch_arxiv_paper_by_id(arxiv_id)
    
    # Handle bioRxiv/medRxiv DOIs
    if doi and doi.startswith("10.1101/"):
        logger.info(f"Detected bioRxiv/medRxiv DOI: {doi}")
        # The doi is already in the correct format for bioRxiv API
        return fetch_biorxiv_paper_by_doi(doi)
    
    # For other DOI formats, we'd need additional API integrations
    logger.warning(f"Unrecognized DOI format or unsupported repository: {doi}")
    return None

def fetch_paper_metadata_from_openalex(doi):
    """
    Fetches paper metadata from OpenAlex using a DOI and creates a ResearchPaper object.
    
    Args:
        doi (str): Digital Object Identifier for the paper
        
    Returns:
        ResearchPaper or None: A ResearchPaper object with the paper metadata, or None if retrieval fails
    """
    try:
        logger.info(f"Fetching metadata from OpenAlex for DOI: {doi}")
        
        # Configure pyalex with an email for responsible API usage
        pyalex.config.email = "user@example.com"  # Replace with a real email in production
        
        # Clean the DOI - OpenAlex doesn't like some prefixes
        if doi.lower().startswith('https://doi.org/'):
            doi = doi[16:]
        
        # Query OpenAlex using the DOI
        works = pyalex.Works().filter(doi=doi).get()
        
        # If we got no results, return None
        if not works or len(works) == 0:
            logger.info(f"No data found in OpenAlex for DOI: {doi}")
            return None
        
        # Get the first work (should be only one for a specific DOI)
        work = works[0]
        
        # Extract authors list
        authors_list = []
        if 'authorships' in work:
            for authorship in work['authorships']:
                if 'author' in authorship and 'display_name' in authorship['author']:
                    authors_list.append(authorship['author']['display_name'])
        
        authors = '; '.join(authors_list)
        
        # Get publication date
        pub_date = work.get('publication_date', '')
        
        # Get corresponding author info (limited in OpenAlex)
        corresponding_author = ""
        corresponding_institution = ""
        
        if 'authorships' in work and len(work['authorships']) > 0:
            # Often the first author is corresponding, but this is not always accurate
            first_authorship = work['authorships'][0]
            if 'author' in first_authorship and 'display_name' in first_authorship['author']:
                corresponding_author = first_authorship['author']['display_name']
            
            # Try to get institution
            if 'institutions' in first_authorship and len(first_authorship['institutions']) > 0:
                if 'display_name' in first_authorship['institutions'][0]:
                    corresponding_institution = first_authorship['institutions'][0]['display_name']
        
        # Get abstract
        abstract = ""
        if 'abstract_inverted_index' in work and work['abstract_inverted_index']:
            # OpenAlex stores abstracts as inverted indices, we need to reconstruct them
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
        
        # Get journal info
        journal_name = ""
        if 'host_venue' in work and work['host_venue'] and 'display_name' in work['host_venue']:
            journal_name = work['host_venue']['display_name']
        
        # Get category/field
        category = ""
        if 'concepts' in work and work['concepts'] and len(work['concepts']) > 0:
            # Use the highest-level concept as category
            concepts = sorted(work['concepts'], key=lambda x: x.get('score', 0), reverse=True)
            if concepts and 'display_name' in concepts[0]:
                category = concepts[0]['display_name']
        
        # Create a ResearchPaper object
        paper = ResearchPaper(
            doi=doi,
            title=work.get('title', ''),
            authors=authors,
            date=pub_date,
            abstract=abstract,
            category=category,
            license="",  # Not typically available in OpenAlex
            version="",  # Not typically available in OpenAlex
            author_corresponding=corresponding_author,
            author_corresponding_institution=corresponding_institution,
            published_journal=journal_name,
            published_date=pub_date,
            published_doi=doi,  # Same as input DOI for published papers
            inclusion_decision=None,
            criteria_assessment=None,
            assessment_explanation=None,
            assessment_datetime=None  # Explicitly set to None to ensure it's blank
        )
        
        logger.info(f"Successfully retrieved metadata from OpenAlex")
        return paper
        
    except Exception as e:
        logger.error(f"Error fetching paper data from OpenAlex: {e}")
        return None

def clean_sentences(sentences):
    """ Clean sentences by removing unwanted patterns and filtering"""
    cleaned = []
    for sentence in sentences:
        # Remove leading/trailing whitespace
        sentence = sentence.strip()

        # Filter out empty or very short sentences
        if len(sentence.split()) < 5:
            continue

        # Remove sentences with too many symbols or very few letters
        alpha_ratio = len(re.findall(r'[a-zA-Z]', sentence)) / (len(sentence) + 1e-5)
        if alpha_ratio < 0.5:
            continue

        # Remove typical metadata patterns
        if re.search(r'(doi|https?://|creativecommons|copyright|journal\.|plos|open access|published|license|funding|competing interests)', sentence, re.IGNORECASE):
            continue

        cleaned.append(sentence)
    return cleaned

def prepare_and_embed_text(text: str, doi: str, embedder) -> List[dict]:
    """
    Cleans, token-clips, and embeds research paper text.
    Returns list of dicts for insertion into DB.
    """
    # 1. Sentence tokenization and cleaning
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = clean_sentences(sentences)
    joined_text = ' '.join(cleaned_sentences)

    # 2. Token-based clipping (final step to make chunks embedding-safe)
    tokenizer = get_tokenizer()
    chunked_documents = tokenizer.split_text(joined_text)

    # 3. Generate embeddings
    embeddings = embedder.embed_documents(chunked_documents)

    results = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunked_documents, embeddings)):
        results.append({
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "doi": doi
        })

    return results

def store_chunks_for_paper(session: Session, paper_id: int, doi: str, embedded_chunks: List[dict]) -> int:
    """
    Stores embedded text chunks into the database for a given research paper.

    Args:
        session: SQLAlchemy session object.
        paper_id: The ID of the research paper these chunks belong to.
        doi: The DOI of the paper.
        embedded_chunks: A list of dicts with keys: 'chunk_text', 'embedding', 'chunk_index'

    Returns:
        The number of chunks inserted.
    """
    try:
        # Look up the paper if paper_id is None
        if paper_id is None:
            # Query the database for the paper
            paper = session.query(ResearchPaper).filter(ResearchPaper.doi == doi).first()
            if paper:
                paper_id = paper.id
            else:
                # Log that we couldn't find the paper
                logger.error(f"No paper found with DOI {doi}, cannot store chunks")
                return False
        existing_chunks = session.query(ResearchPaperChunk.chunk_index)\
                        .filter(ResearchPaperChunk.paper_id == paper_id)\
                        .all()
        existing_indexes = {chunk[0] for chunk in existing_chunks}


        # Now insert the chunks with the valid paper_id
        count = 0
        for chunk in embedded_chunks:
            # Skip if this chunk index already exists
            if chunk['chunk_index'] in existing_indexes:
                logger.debug(f"Skipping existing chunk {chunk['chunk_index']} for paper {paper_id}")
                continue
        for chunk in embedded_chunks:
            chunk_obj = ResearchPaperChunk(
                paper_id=paper_id,
                doi=doi,
                chunk_index=chunk['chunk_index'],
                chunk_text=chunk['chunk_text'],
                embedding=chunk['embedding'],
                section=chunk.get('section')
            )
            session.add(chunk_obj)
            count += 1
        session.commit()
        return count
    
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        session.rollback()
        return 0
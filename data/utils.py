# data/utils.py

import httpx
import re
import zlib
import logging
from data.db import get_db_session, close_db_session
from data.models import AbstractGenerationExample

logger = logging.getLogger(__name__)

def make_api_request(url, description="API"):
    """Make an HTTPX request with error handling and return JSON response."""
    try:
        logger.debug(f"Fetching data from: {url}")
        response = httpx.get(url)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"Error fetching {description}: {e}")
        return None
    

def format_modified_abstract(original_abstract: str, modified_abstract: str) -> str:
    """
    Compare original and modified abstracts and format the differences using the
    required [[original text, modified text]] notation.
    
    This is a best-effort approach that will likely require human review to ensure
    the modifications are properly marked according to the guidelines.
    
    Args:
        original_abstract (str): The original abstract text
        modified_abstract (str): The human-modified abstract text
        
    Returns:
        str: A formatted string showing modifications with [[original, modified]] notation
    """
    import difflib
    
    # Use difflib to find differences between words
    original_words = original_abstract.split()
    modified_words = modified_abstract.split()
    
    matcher = difflib.SequenceMatcher(None, original_words, modified_words)
    
    # Build the formatted result
    result_parts = []
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_segment = ' '.join(original_words[i1:i2]) if i1 < i2 else ''
        mod_segment = ' '.join(modified_words[j1:j2]) if j1 < j2 else ''
        
        if op == 'equal':
            # Unchanged text - add as is
            result_parts.append(orig_segment)
        else:
            # Text was changed - use bracket notation
            if orig_segment or mod_segment:
                result_parts.append(f"[[{orig_segment}, {mod_segment}]]")
    
    return ' '.join(result_parts)


def add_abstract_generation_example(research_paper_id, domain, example_abstract):
    """
    Adds a new AbstractGenerationExample to the database. AbstractGenerationExample are used to populate
    the abstract_modification_prompt templates with examples of how the LLM should modify the abstract. 
    
    Args:
        research_paper_id (int): ID of the related research paper
        domain (str): The scientific domain (e.g., 'neuroscience', 'biotechnology')
        example_abstract (str): The modified abstract example with [[original, modified]] formatting
        
    Returns:
        AbstractGenerationExample: The created example object with ID populated, or None if there was an error
    """
    session = get_db_session()
    try:
        # Create new example
        example = AbstractGenerationExample(
            research_paper_id=research_paper_id,
            domain=domain,
            example_abstract=example_abstract
        )
        
        # Add to session and commit
        session.add(example)
        session.commit()
        
        logger.info(f"Successfully added example for research paper ID {research_paper_id}")
        return example
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding abstract generation example: {e}")
        return None
    finally:
        close_db_session(session)


def extract_abstract_pair(abstract):
    """
    Extract original and incorrect versions of an abstract from a bracketed format.
    
    This function parses abstracts containing modifications in the format [[original, modified]],
    where each bracketed section represents a change between the original (correct) and 
    modified (incorrect) versions. The function returns both versions as separate strings.
    
    Args:
        abstract (str): The abstract text containing modifications in [[original, modified]] format
    
    Returns:
        tuple: A pair of (original_abstract, incorrect_abstract) where:
            - original_abstract contains only the correct text
            - incorrect_abstract contains the modified/incorrect text
    
    Raises:
        ValueError: If a bracketed section contains multiple commas, making it ambiguous
                   which parts represent original vs. modified text
    
    Example:
        >>> text = "The neurons [[increased, decreased]] activity during the task."
        >>> extract_abstract_pair(text)
        ("The neurons increased activity during the task.", 
         "The neurons decreased activity during the task.")
    """
    # Find all bracketed modifications using regex pattern
    # Each match will be the content inside the brackets without the brackets themselves
    matches = re.findall(r'\[\[(.*?)\]\]', abstract)
    
    # Create copies of the original abstract to build both versions
    original_abstract = abstract
    incorrect_abstract = abstract
    
    # Process each bracketed modification
    for match in matches:
        # Check for ambiguous parsing cases (multiple commas)
        if match.count(',') > 1:
            raise ValueError(
                f"Multiple commas found in bracketed section: [[{match}]]. " 
                "This makes it ambiguous which parts represent original vs. modified text."
            )
        
        # Split the match into original and modified text segments
        items = match.split(',')
        
        # Ensure we have exactly two items (original and modified)
        if len(items) != 2:
            raise ValueError(
                f"Expected exactly one comma in bracketed section: [[{match}]]"
            )
        
        # Replace the bracketed section with the appropriate text in each version
        # For original version: use the first item (items[0])
        original_abstract = original_abstract.replace('[[' + match + ']]', items[0].strip())
        
        # For incorrect version: use the second item (items[1])
        incorrect_abstract = incorrect_abstract.replace('[[' + match + ']]', items[1].strip())

    return original_abstract, incorrect_abstract

def zlib_compression_size(text: str) -> int:
    """
    Compute the size of the compressed text using zlib compression.
    Used to estimate the entropy of the text and in to compute the perplexity-zlib ratio,
    which is a metric used to evalute whether the model is memorizing the text.
    
    Args:
        text (str): The text to compress
    
    Returns:
        int: The size of the compressed text in bytes
    """
    return len(zlib.compress(bytes(text, 'utf-8')))

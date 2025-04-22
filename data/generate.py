#data/generate.py

import json
import jinja2
import datetime
import re
import os
import sys
import logging


from prompts_library.data_creation import neuroscience_inclusion_assessment_prompt, economics_inclusion_assessment_prompt
from data.db import get_db_session, close_db_session
from data.models import BrainbenchInclusionResponse, FalseAbstractGenerationEvent, ResearchPaper, AbstractGenerationExample
from data.utils import extract_abstract_pair
from services.llm_services import BasicOpenAI
from services.arxiv import arxiv_taxonomy
from typing import List, Optional, Tuple
from sqlalchemy import func, or_

logger = logging.getLogger(__name__)

class AbstractModifier:
    """Class for creating incorrect versions of research paper abstracts."""
    
    def __init__(self, prompt_template: str, domain: str, modified_abstract_1: str = None, modified_abstract_2: str = None, llm_client: Optional[BasicOpenAI] = None):
        """
        Initialize the AbstractModifier with required prompt template and domain.
        
        Args:
            prompt_template: Template to use for generating incorrect abstracts
            domain: Scientific domain of the paper (e.g., "neuroscience")
            llm_client: OpenAI client instance (will create one using GPT-4 if None)
        """
        self.modification_attempts = 3
        self.prompt_template = prompt_template
        self.domain = domain
        self.llm_client_model = "gpt-4o"
        
        # Set Example modified abstracts for the prompt
        # If not provided, query the database for examples
        self.modified_abstract_1 = modified_abstract_1
        self.modified_abstract_2 = modified_abstract_2

        if not self.modified_abstract_1 or not self.modified_abstract_2:
            self._retrieve_abstract_examples()
        
        # Create the LLM client if not provided, using environment variables
        if llm_client is None:
            self.llm_client = BasicOpenAI(model=self.llm_client_model)
        else:
            self.llm_client = llm_client

    def _retrieve_abstract_examples(self):
        """
        Retrieve examples of modified abstracts from the database.
        
        Returns:
            Tuple of two modified abstracts
        """
        session = get_db_session()
        # Query the database for two examples of modified abstracts in the self.domain
        examples = session.query(AbstractGenerationExample)\
            .filter(func.lower(AbstractGenerationExample.domain) == func.lower(self.domain))\
                .limit(2).all()
        
        if len(examples) < 2:
            raise ValueError("Not enough modified abstract examples found in the database.")
        
        self.modified_abstract_1 = examples[0].example_abstract
        self.modified_abstract_2 = examples[1].example_abstract

        return examples[0].example_abstract, examples[1].example_abstract
    
    def _cleanse_modified_abstract(self, raw_modified_abstract: str, original_abstract: str):
        """
        Clean and validate the LLM-generated modified abstract to ensure it properly 
        follows the [[original, modified]] format and maintains consistency with the 
        original abstract's text.
        
        This function:
        1. Verifies that the modified abstract contains modification brackets
        2. Ensures the text outside the modification brackets matches the original abstract
        3. Handles prefix consistency (text before the first modification)
        4. Handles suffix consistency (text after the last modification)
        5. Rebuilds the abstract if necessary to maintain alignment with the original

        Args:
            raw_modified_abstract (str): The unprocessed modified abstract from the LLM
            original_abstract (str): The original abstract text for comparison
            
        Returns:
            str: A cleaned and properly formatted modified abstract
            
        Raises:
            ValueError: If the modified abstract lacks proper formatting or cannot
                       be aligned with the original abstract
        """
        def normalize_text(text):
            # Normalize spaces
            text = re.sub(r"\s+", " ", text)
            # Normalize dashes
            text = re.sub(r"[-—]", "-", text)
            # Normalize ellipses
            text = text.replace('…', '...')

            text = re.sub(r'[\u00B0\u00BA\u2070]', ' degrees ', text)  # Degree symbols
            text = re.sub(r'[\u00B5\u03BC]', 'u', text)  # Micro symbols
            text = re.sub(r'[\u00B1]', ' plus or minus ', text)  # Plus-minus symbols
            text = re.sub(r'[\u00D7]', ' times ', text)  # Multiplication symbols
            text = re.sub(r'[\u00F7]', ' divided by ', text)  # Division symbols
            return text.strip()
           
        normalized_modified_abstract = normalize_text(raw_modified_abstract)
        normalized_original_abstract = normalize_text(original_abstract)
        
        # throw error if the normalized_modified_abstract strarts with brackets
        if normalized_modified_abstract.startswith("[["):
            raise ValueError("The generated abstract starts with a modification bracket.")
        
        # Find the first modification bracket
        bracket_index = normalized_modified_abstract.find("[[")
        
        if (bracket_index == -1):
            raise ValueError("No modification brackets found in the generated abstract.")
        
        # Extract everything before the bracket
        modified_prefix = normalized_modified_abstract[:bracket_index]

        # Compare with the original abstract's beginning
        # remove text not shared by both
        if not normalized_original_abstract.startswith(modified_prefix):

            # iterate backward through all toks in the prefix until the pattern 
            # is no longer found in the original abstract. This will give us the correct prefix
            prefix_toks = modified_prefix.split()
            original_prefix =  prefix_toks[-1]
            for prefix_tok in reversed(prefix_toks[:-1]):
                if prefix_tok + " " + original_prefix not in normalized_original_abstract:
                    break
                original_prefix = prefix_tok + " " + original_prefix

            # additional check to ensure the prefix actually starts the original abstract
            if not normalized_original_abstract.startswith(original_prefix):
                raise ValueError("The generated abstract prefix does not match the original abstract.")

            # replace the prefix with the origninal prefix
            normalized_modified_abstract = original_prefix + normalized_modified_abstract[len(modified_prefix):]

        # Handle the suffix (everything after the last modification bracket)
        last_bracket_index = normalized_modified_abstract.rfind("]]")
        
        if last_bracket_index == -1:
            raise ValueError("No closing modification brackets found in the generated abstract.")
        
        # Get the suffix (everything after the last "]]")
        modified_suffix = normalized_modified_abstract[last_bracket_index + 2:].strip()
        
        # If there's a suffix, check if it matches the end of the original abstract
        if modified_suffix:
            if not normalized_original_abstract.endswith(modified_suffix):
                # Similar approach to prefix but now going forward through tokens
                suffix_tokens = modified_suffix.split()
                original_suffix = suffix_tokens[0]
                
                for suffix_token in suffix_tokens[1:]:
                    test_suffix = original_suffix + " " + suffix_token
                    if test_suffix not in normalized_original_abstract:
                        break
                    original_suffix = test_suffix
                
                # additional check to ensure the suffix actually ends the original abstract
                if not normalized_original_abstract.endswith(original_suffix):
                    raise ValueError("The generated abstract suffix does not match the original abstract.")
                
                # Replace the suffix with the original suffix found
                normalized_modified_abstract = normalized_modified_abstract[:last_bracket_index + 2] + " " + original_suffix
        
        return normalized_modified_abstract.strip()
    
    def modify_abstract(self, paper: ResearchPaper, save_event: bool = True) -> Tuple[ResearchPaper, Optional[FalseAbstractGenerationEvent]]:
        """
        Generate an incorrect abstract, update the paper object, and record the generation event.
        
        Args:
            paper: A ResearchPaper object to modify
            save_event: Whether to save the generation events to the database
            
        Returns:
            Tuple containing:
                - The updated ResearchPaper object with gpt4_incorrect_abstract populated
                - The FalseAbstractGenerationEvent of the successful generation, or None
        """
        
        def begins_and_ends_same(original, modified, matching_toks = 4):
            """
            This function tests if the first and last matching_toks tokens of the original and modified text are the same.
            """
            original_toks = original.split()
            modified_toks = modified.split()
            return (len(original_toks) >= matching_toks and len(modified_toks) >= matching_toks and
                    original_toks[:matching_toks] == modified_toks[:matching_toks] and 
                    original_toks[-matching_toks:] == modified_toks[-matching_toks:])
        
        # Format the prompt with paper data
        template = jinja2.Template(self.prompt_template)
        formatted_prompt = template.render(
            example_1 = self.modified_abstract_1,
            example_2 = self.modified_abstract_2,
            domain = self.domain,
            abstract_to_edit = paper.abstract
        )
        
        # Generate the incorrect abstract
        successful_event = None
        
        try:
            attempts = 0
            while attempts < self.modification_attempts:
                attempts += 1
                logger.info(f"Attempt {attempts}/{self.modification_attempts} to generate false abstract")
                
                # Generate response from LLM
                raw_response = self.llm_client.complete(
                    prompt=formatted_prompt,
                    system_message="You are a helpful research assistant.",
                    temperature=0.7,  # Higher temperature for creative variations
                    max_tokens=None
                )
                
                # Extract the raw response
                raw_modified_abstract = raw_response.strip()
                
                # Always record the generation event regardless of success
                generation_event = self._record_generation_event(
                    research_paper_id=paper.id,
                    model=self.llm_client_model,
                    raw_response=raw_response,
                    generated_abstract=raw_modified_abstract
                )
                
                # Initial check - can we extract original/modified pair directly?
                try:
                    rebuilt_original_abstract, rebuilt_modified_abstract = extract_abstract_pair(raw_modified_abstract)
                    
                    # Check if the abstracts match at beginning and end
                    if begins_and_ends_same(paper.abstract, rebuilt_modified_abstract):
                        logger.info("Direct extraction successful.")
                        #Update the paper
                        paper.gpt4_incorrect_abstract = rebuilt_modified_abstract
                        successful_event = generation_event
                        return paper, successful_event
                except Exception as extract_error:
                    logger.debug(f"Initial extraction failed: {extract_error}")
                
                # Initial extraction didn't work or abstracts didn't match, try cleansing
                try:
                    logger.debug("Attempting to cleanse modified abstract...")
                    cleansed_abstract = self._cleanse_modified_abstract(raw_modified_abstract, paper.abstract)
                    
                    # Extract from the cleansed abstract
                    rebuilt_original_abstract, rebuilt_modified_abstract = extract_abstract_pair(cleansed_abstract)
                    
                    # Check if the cleansed abstracts match at beginning and end
                    if begins_and_ends_same(paper.abstract, rebuilt_modified_abstract):
                        logger.info("Cleansing successful.")
                        # Update the paper
                        paper.gpt4_incorrect_abstract = rebuilt_modified_abstract
                        successful_event = generation_event
                        return paper, successful_event
                    else:
                        logger.debug("Cleansing did not yield matching abstracts.")
                except Exception as cleanse_error:
                    logger.debug(f"Cleansing failed: {cleanse_error}")
                
                # If we got here, both direct extraction and cleansing failed
                logger.debug("This attempt did not produce a usable abstract")
            
            # If we've exhausted all attempts without success
            logger.error(f"Failed to generate a valid modified abstract after {self.modification_attempts} attempts")
            return paper, None
            
        except Exception as e:
            logger.error(f"Error generating incorrect abstract: {e}")
            return paper, None
        
    def _record_generation_event(self, research_paper_id: int, model: str, 
                               raw_response: str, generated_abstract: str) -> FalseAbstractGenerationEvent:
        """
        Record a false abstract generation event in the database.
        
        Args:
            research_paper_id: ID of the research paper
            model: Name of the LLM model used
            raw_response: Complete raw response from the model
            generated_abstract: Processed/extracted incorrect abstract
            
        Returns:
            The created FalseAbstractGenerationEvent
        """
        session = get_db_session()
        try:
            # Create new event
            event = FalseAbstractGenerationEvent(
                research_paper_id=research_paper_id,
                model=model,
                raw_response=raw_response,
                generated_abstract=generated_abstract,
                created_at=datetime.datetime.utcnow()
            )
            
            # Add to session and commit
            session.add(event)
            session.commit()
            
            # Return the created event with ID populated
            return event
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording generation event: {e}")
            return None
        finally:
            close_db_session(session)
  
def assess_paper_for_inclusion(paper: ResearchPaper, llm_client: Optional[BasicOpenAI] = None, domain: str = "neuroscience") -> ResearchPaper:
    """
    Assesses a ResearchPaper object against the BrainBench inclusion criteria for a specific domain.

    Args:
        paper: A ResearchPaper object to assess
        llm_client: An initialized BasicOpenAI client (will create one if None)
        domain: The scientific domain (e.g., "neuroscience", "economics") to use for assessment prompts.

    Returns:
        The updated ResearchPaper object or None if assessment fails.
    """

    # Initialize BasicOpenAI client if not provided
    if llm_client is None:
        llm_client = BasicOpenAI()

    assessment_prompt = None
    # Select the appropriate prompt based on domain
    # Ensure domain matching is case-insensitive for robustness
    domain_lower = domain.lower()
    if domain_lower == "economics":
        assessment_prompt = economics_inclusion_assessment_prompt
    elif domain_lower == "neuroscience": # Assuming neuroscience is the default or another option
        assessment_prompt = neuroscience_inclusion_assessment_prompt
    else:
        # Default or raise an error if domain is unsupported
        logger.warning(f"Unsupported domain '{domain}'. Using default neuroscience prompt.")
        assessment_prompt = neuroscience_inclusion_assessment_prompt
        # Alternatively: raise ValueError(f"Unsupported domain for assessment: {domain}")

    if not assessment_prompt:
         logger.error(f"No assessment prompt found for domain '{domain}'.")
         return None # Or raise an error

    # Format the prompt with paper data using jinja2
    template = jinja2.Template(assessment_prompt)
    formatted_prompt = template.render(
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        domain=domain
    )

    # Query the OpenAI API via our wrapper class
    try:
        # Use the BasicOpenAI wrapper to make the API call
        result_json = llm_client.complete(
            prompt=formatted_prompt,
            system_message="You are a helpful research assistant.",
            temperature=0.0,
            max_tokens=800,
            json_response=True
        )

        # Convert to our Pydantic model
        result = BrainbenchInclusionResponse.model_validate(result_json)

        # Update the paper object
        paper.inclusion_decision = result.decision.lower() == "include"

        # Convert Pydantic models to dict to ensure JSON serialization works
        criteria_dict = {}
        for key, criterion in result.criteria_assessments.items():
            criteria_dict[key] = criterion.model_dump()

        paper.criteria_assessment = criteria_dict
        paper.assessment_explanation = result.explanation
        paper.assessment_datetime = datetime.datetime.utcnow()

        return paper

    except Exception as e:
        logger.error(f"Error assessing paper ID {paper.id}: {e}")
        # Optionally rollback session if changes were staged before exception
        # session = Session.object_session(paper)
        # if session:
        #     session.rollback()
        return None

def assess_papers_by_category(domain: str, categories: List[str], llm_client: Optional[BasicOpenAI] = None, limit: int = 10):
    """
    Assesses unreviewed papers from specific categories for inclusion based on the given domain.

    Args:
        domain: The scientific domain (e.g., "neuroscience", "economics") for assessment.
        categories: List of arXiv category strings to filter papers.
        llm_client: An initialized BasicOpenAI client (will create one if None).
        limit: Maximum number of papers to assess.
    """
    # Get database session
    session = get_db_session()

    # Initialize BasicOpenAI client if not provided
    if llm_client is None:
        llm_client = BasicOpenAI()

    try:
        # Find papers in the specified categories that haven't been assessed yet
        papers = session.query(ResearchPaper).filter(
            or_(
                *[ResearchPaper.category.contains(cat) for cat in categories]
            ),
            ResearchPaper.inclusion_decision.is_(None),
            ResearchPaper.abstract.isnot(None),
            func.length(ResearchPaper.abstract) > 0
        ).order_by(func.random()).limit(limit).all()

        if not papers:
            logger.info(f"No unassessed papers found in the specified categories for domain '{domain}'.")
            return

        logger.info(f"\nFound {len(papers)} papers in domain '{domain}' to assess.")

        # Process each paper
        processed_count = 0
        for i, paper in enumerate(papers, 1):
            logger.info(f"\nProcessing {domain} paper {i}/{len(papers)} (ID: {paper.id}):")
            logger.info(f"Title: {paper.title}")
            logger.info(f"Categories: {paper.category}") # Helpful to see the category

            # Assess the paper with the specified domain
            updated_paper = assess_paper_for_inclusion(paper, llm_client, domain=domain)

            if updated_paper:
                # Save changes to database after each paper
                # The session associated with updated_paper should handle the commit
                session.commit()
                processed_count += 1

                # Display assessment results
                decision_str = "Include" if updated_paper.inclusion_decision else "Exclude"
                logger.info(f"Decision: {decision_str}")
                logger.info(f"Explanation: {updated_paper.assessment_explanation[:200]}...")
            else:
                logger.warning(f"Assessment failed for paper ID {paper.id}")
                # No commit needed if assessment failed, potentially rollback if needed elsewhere

        logger.info(f"\nFinished assessing {domain} papers. Processed {processed_count}/{len(papers)} successfully.")

    except Exception as e:
        logger.error(f"An error occurred during the assessment process for domain '{domain}': {e}")
        session.rollback() # Rollback any potential partial changes in case of error
    finally:
        close_db_session(session)

def generate_abstracts_for_papers(papers: List, domain: str, generation_prompt_template: str, llm_client: Optional[BasicOpenAI] = None) -> Tuple[int, int]:
    """
    Generates false abstracts for a list of research papers.

    Args:
        papers (List[ResearchPaper]): List of research paper objects.
        domain (str): Scientific domain for the abstracts.
        generation_prompt_template (str): Template to use for abstract generation.
        llm_client (Optional[BasicOpenAI]): An optional LLM client. If not provided, one will be created.

    Returns:
        Tuple[int, int]: A tuple containing the number of papers processed and the number of successful modifications.
    """
    logger = logging.getLogger(__name__)
    processed_count = 0
    success_count = 0

    if llm_client is None:
        llm_client = BasicOpenAI()

    from data.generate import AbstractModifier
    modifier = AbstractModifier(
        prompt_template=generation_prompt_template,
        domain=domain,
        llm_client=llm_client
    )

    for paper in papers:
        logger.info(f"Processing paper with ID {paper.id}")
        try:
            updated_paper, generation_event = modifier.modify_abstract(paper, save_event=True)
            processed_count += 1
            if updated_paper.gpt4_incorrect_abstract:
                success_count += 1
                logger.info(f"Success: Generated false abstract for paper ID {paper.id}")
            else:
                logger.warning(f"Warning: No false abstract generated for paper ID {paper.id}")
        except Exception as e:
            logger.error(f"Error processing paper ID {paper.id}: {e}")

    logger.info(f"Processed {processed_count} papers, with {success_count} successes in abstract generation")
    return processed_count, success_count

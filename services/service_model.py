from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod
from data.models import ResearchPaper
from services.utils import extract_structured_text, get_fallback_text

logger = logging.getLogger(__name__)

class ServiceQueryInfo(BaseModel):
    """
    Metadata description for a research paper retrieval service used in the RAG system.

    This model informs the LLM query generation process and query dispatch logic about
    the nature and capabilities of the service. It helps determine how to phrase search queries
    and which services are relevant for a given abstract or topic.

    Attributes:
        name (str):
            The human-readable name of the service (e.g., "PubMed").

        description (str):
            A brief summary of the service, its focus domain, and how it can be queried.

        domains (List[str]):
            The fields of study this service supports (e.g., neuroscience, economics).
            Used to determine if a service is relevant for a particular abstract.

        query_format (Literal["keyword", "boolean", "natural"]):
            Indicates the expected style of query input:
                - "keyword": Space-separated keywords or short phrases.
                - "boolean": Boolean operators and/or field-specific syntax allowed.
                - "natural": Full natural language questions or descriptions.

        search_fields (List[str]):
            Which parts of the paper can be searched (e.g., title, abstract, full text).
            Useful for guiding prompt construction.

        enabled (bool, optional):
            Whether this service is currently active and should be considered for query generation.
            Defaults to True.
    """
    name: str
    description: str
    domains: List[str]
    query_format: Literal["keyword", "boolean", "natural"]
    search_fields: List[str]
    enabled: bool = True


class BasePaperService(ABC):
    """
    Abstract base class for paper retrieval services.
    
    This class provides a common interface and shared functionality for all 
    paper service implementations. Services should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def fetch_paper_metadata(self, identifier: str) -> Dict[str, Any]:
        """
        Fetch metadata for a paper using its identifier (e.g., DOI, ID).
        
        Args:
            identifier (str): The identifier for the paper
            
        Returns:
            Dict[str, Any]: Dictionary containing paper metadata
        """
        pass
    
    def fetch_full_text(self, paper: ResearchPaper) -> str:
        """
        Fetch the full text or enhanced metadata for a paper.
        
        This default implementation:
        1. Attempts to extract a valid identifier
        2. Fetches metadata using fetch_paper_metadata
        3. Formats the metadata into structured text
        
        Args:
            paper (ResearchPaper): The paper to fetch text for
            
        Returns:
            str: Structured text containing paper content
        """
        try:
            # Extract identifier (implemented by subclasses)
            identifier = self.extract_identifier(paper.doi)
            
            if not identifier:
                logger.warning(f"Could not extract valid identifier from DOI: {paper.doi}")
                return get_fallback_text(paper)
                
            # Fetch metadata
            metadata = self.fetch_paper_metadata(identifier)
            
            if not metadata:
                logger.warning(f"Could not fetch metadata for identifier: {identifier}")
                return get_fallback_text(paper)
                
            # Format text
            return self.format_paper_text(metadata)
            
        except Exception as e:
            logger.error(f"Error in fetch_full_text: {e}")
            return get_fallback_text(paper)
    
    @abstractmethod
    def extract_identifier(self, doi: str) -> Optional[str]:
        """
        Extract a service-specific identifier from a DOI.
        
        Args:
            doi (str): The DOI to extract an identifier from
            
        Returns:
            Optional[str]: The extracted identifier or None if not found
        """
        pass
    
    def format_paper_text(self, metadata: Dict[str, Any]) -> str:
        """
        Format paper metadata into structured text.
        
        Args:
            metadata (Dict[str, Any]): Dictionary containing paper metadata
            
        Returns:
            str: Structured text representation of the paper
        """
        # Create a flattened dictionary of components
        components = {}
        
        # Common metadata fields to include
        common_fields = [
            "title", "abstract", "authors", "date", "journal", 
            "doi", "text", "category"
        ]
        
        # Extract common fields
        for field in common_fields:
            if field in metadata and metadata[field]:
                components[field] = metadata[field]
        
        # Extract any additional fields
        for key, value in metadata.items():
            if key not in common_fields and value and isinstance(value, (str, int, float)):
                components[key] = str(value)
        
        # Use utility function to create structured text
        return extract_structured_text(None, components)
    
    @abstractmethod
    def run_queries(self, queries: List[str], max_results: int = 10) -> List[ResearchPaper]:
        """
        Execute queries against the service.
        
        Args:
            queries (List[str]): List of search queries
            max_results (int): Maximum number of results per query
            
        Returns:
            List[ResearchPaper]: List of research paper objects
        """
        pass

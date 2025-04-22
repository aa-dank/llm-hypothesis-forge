# data/models.py
# Note that this file is not for representations of llm models

import json
import datetime
from typing import List, Dict, Optional, Any

from pydantic import BaseModel
from sqlalchemy import Column, Float, Integer, String, Boolean, Text, DateTime, Index, UniqueConstraint, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, TEXT
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class JSONType(TypeDecorator):
    """
    Custom SQLAlchemy type decorator for handling JSON data.
    
    This type automatically handles the conversion between Python 
    dictionaries/lists and JSON strings when storing and retrieving
    data from the database.
    
    When data is being stored in the database, Python objects are 
    serialized to JSON strings. When queried from the database,
    JSON strings are deserialized back to Python objects.
    """
    impl = TEXT
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            if type(value) == dict:
                return value
            return json.loads(value)
        return None


class CriterionAssessment(BaseModel):
    """
    This class represents one of each of the assessment crtiteria of a research paper being assessed against the BrainBench inclusion criteria.
    """
    met: bool
    explanation: str


class BrainbenchInclusionResponse(BaseModel):
    """
    This class represents the response of the BrainBench inclusion assessment of a research paper.
    """
    decision: str
    # criteria_assessments is a dictionary where the key is the criterion name and the value is a CriterionAssessment object
    criteria_assessments: Dict[str, CriterionAssessment]
    explanation: str


class ResearchPaper(Base):
    """
    This class represents a research paper in the research database.
    """
    
    __tablename__ = 'research_papers'
    
    id = Column(Integer, primary_key=True)
    doi = Column(Text, unique=True, index=True)  # Added unique constraint and index
    title = Column(String)
    authors = Column(String)
    date = Column(String)
    abstract = Column(Text)
    category = Column(String)
    license = Column(String)
    version = Column(String)
    author_corresponding = Column(String)
    author_corresponding_institution = Column(String)
    published_journal = Column(String)
    published_date = Column(String)
    published_doi = Column(String)
    # This field is used to store the inclusion decision made by the LLM given the criteria prompt
    inclusion_decision = Column(Boolean)
    # This field is used to store the assessment of the paper against the BrainBench inclusion criteria
    criteria_assessment = Column(JSONType)
    # This field is used to store the datetime of the assessment
    assessment_datetime = Column(DateTime)
    # This field is used to store the explanation of the inclusion assessment
    assessment_explanation = Column(Text)
    # This feature serves to track which papers were previously in the BrainBench datasets
    previously_in_brainbench = Column(Boolean, default=False)
    # columns for incorrect abstracts
    human_incorrect_abstract = Column(Text)
    gpt4_incorrect_abstract = Column(Text)
    contents = relationship("ResearchPaperChunk", back_populates="paper", cascade="all, delete-orphan")
    
    # Add a unique constraint (additional to column-level constraint)
    __table_args__ = (
        UniqueConstraint('doi', name='uix_research_papers_doi'),
    )
    

    def to_dict(self, exclude=None):
        """
        Convert the ResearchPaper object to a dictionary.
        
        Args:
            exclude: Optional list of field names to exclude from the dictionary.
                   
        Returns:
            dict: Dictionary representation of the ResearchPaper object
        """
        if exclude is None:
            exclude = []
            
        # Get all column attributes automatically
        result = {c.name: getattr(self, c.name) 
                 for c in self.__table__.columns 
                 if c.name not in exclude}
        
        return result
    
    def __repr__(self):
        return f"<ResearchPaper(id={self.id}, title='{self.title[:30] + '...' if len(self.title) > 30 else self.title}')>"
    

class AbstractGenerationExample(Base):
    """
    This class represents examples of abstracts modified by the LLM to be used for testing.
    The examples are used to populate the abstract_modification_prompt templates with examples of how the LLM 
    should modify the abstract.
    """
    __tablename__ = 'abstract_generation_examples'

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_paper_id = Column(Integer, index=True)
    domain = Column(String)
    example_abstract = Column(Text)


class FalseAbstractGenerationEvent(Base):
    """
    Class used for representing the metadata about calls to LLM service to generate false abstracts.
    """
    __tablename__ = 'false_abstract_generation_events'

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_paper_id = Column(Integer, index=True)
    model = Column(String)
    raw_response = Column(Text)
    generated_abstract = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class PerplexityScoreEvent(Base):
    """
    Store perplexity scores for model evaluations on abstracts with specific prompt templates.
    """
    __tablename__ = 'perplexity_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    # Link to research paper if applicable (could be NULL for external test cases)
    research_paper_id = Column(Integer, index=True, nullable=True)
    
    # Model information
    model = Column(String, index=True)  # e.g., "gpt2-medium_scratch_neuro_tokenizer"
    
    # Text being evaluated
    abstract_text = Column(Text)  # Store the actual text being evaluated
    
    # Source categorization of the abstract
    abstract_source = Column(String, nullable=False)  # Values: 'original', 'gpt4', 'human'

    # Prompt information
    prompt_template_name = Column(String)  # e.g., "gpt2-medium_scratch_neuro_tokenizer"
    full_prompt = Column(Text)  # The complete prompt with abstract inserted
    
    # Results
    zlib_compression_size = Column(Integer)
    perplexity_score = Column(Float)
    zlib_perplexity_ratio = Column(Float)
    
    # Metadata
    evaluation_datetime = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Optional additional fields
    additional_metadata = Column(JSONType, nullable=True)
    
    __table_args__ = (
        # Create indexes for common query patterns
        Index('idx_model_paper', model, research_paper_id),
        Index('idx_abstract_source', abstract_source),
    )


class PerplexityPromptTemplate(Base):
    """
    Store the prompt templates used for calculating perplexity scores.
    """
    __tablename__ = 'perplexity_prompt_templates'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model = Column(String, index=True)
    domain = Column(String, index=True)
    prompt_template = Column(Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model': self.model,
            'domain': self.domain
        }


class PaperCitation(Base):
    """
    This class represents citation information for research papers.
    """
    
    __tablename__ = 'paper_citations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doi = Column(Text)
    title = Column(Text)
    year = Column(Float)
    venue = Column(Text)
    citationCount = Column(Integer)
    paperId = Column(Text)
    url = Column(Text)
    cited_paper_doi = Column(Text)
    
    def __repr__(self):
        return f"<PaperCitation(doi='{self.doi}', title='{self.title[:30] + '...' if self.title and len(self.title) > 30 else self.title}')>"
    

class ResearchPaperChunk(Base):
    """
    Represents a single text chunk from a research paper, such as a paragraph or section fragment.
    
    Each chunk is associated with a parent research paper via `paper_id` and stores:
    - its position within the paper (`chunk_index`),
    - the chunked text itself (`chunk_text`),
    - the embedding for similarity search or retrieval tasks (`embedding`),
    - optional metadata such as section name and DOI,
    - creation timestamp for logging or auditing purposes.

    A uniqueness constraint ensures that no two chunks from the same paper can have the same index.
    """

    __tablename__ = 'research_paper_chunks'

    # Ensure that each (paper_id, chunk_index) pair is unique
    __table_args__ = (
        UniqueConstraint('paper_id', 'chunk_index', name='uix_paper_chunk'),
    )

    id = Column(Integer, primary_key=True)  # Unique identifier for the chunk (autoincremented)
    
    paper_id = Column(
        Integer,
        ForeignKey('research_papers.id', ondelete='CASCADE'),
        nullable=False
    )  # Foreign key linking to the parent paper

    doi = Column(Text, nullable=False)  # DOI of the paper (can help with quick filtering)

    section = Column(Text, nullable=True)  # Optional: section of the paper (e.g. 'Introduction')

    chunk_index = Column(Integer, nullable=False)  # Position of the chunk within the paper

    chunk_text = Column(Text, nullable=False)  # The actual chunked text content

    embedding = Column(Vector(1536), nullable=False)  # Embedding vector (e.g. from OpenAI ada-002)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )  # Timestamp when the chunk was created

    # Define relationship to parent research paper
    paper = relationship("ResearchPaper", back_populates="contents")
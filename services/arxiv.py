import re
import urllib
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from data.utils import make_api_request
from data.models import ResearchPaper
from services.service_model import ServiceQueryInfo, BasePaperService
from services.utils import pdf_to_text

logger = logging.getLogger(__name__)

service_info = ServiceQueryInfo(
    name="arXiv",
    description="An open-access repository of preprints in fields such as physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics.",
    domains=["neuroscience", "computer science", "economics"],
    query_format="boolean",
    search_fields=["title", "abstract", "full text"],
    enabled=True
)

# arxiv_taxonomy represents the taxonomy of arXiv categories per 
# https://arxiv.org/category_taxonomy
arxiv_taxonomy = {
    "Computer Science": {
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
    },
    "Economics": {
        "econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics",
    },
    "Electrical Engineering and Systems Science": {
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
    },
    "Mathematics": {
        "math.AC": "Commutative Algebra",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History and Overview",
        "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology",
        "math.LO": "Logic",
        "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings and Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
    },
    "Physics": {
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis, Statistics and Probability",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics",
        "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics and Society",
        "physics.space-ph": "Space Physics",
    },
    "Quantitative Biology": {
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues and Organs",
    },
    "Quantitative Finance": {
        "q-fin.CP": "Computational Finance",
        "q-fin.EC": "Economics",
        "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.PR": "Pricing of Securities",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading and Microstructure",
    },
    "Statistics": {
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Theory",
    },
    "Physics": {
        "astro-ph": "Astrophysics",
        "astro-ph.GA": "Astrophysics of Galaxies",
        "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
        "astro-ph.EP": "Earth and Planetary Astrophysics",
        "astro-ph.HE": "High Energy Astrophysical Phenomena",
        "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
        "astro-ph.SR": "Solar and Stellar Astrophysics",
        "cond-mat": "Condensed Matter",
        "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
        "cond-mat.mtrl-sci": "Materials Science",
        "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
        "cond-mat.other": "Other Condensed Matter",
        "cond-mat.quant-gas": "Quantum Gases",
        "cond-mat.soft": "Soft Condensed Matter",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "cond-mat.str-el": "Strongly Correlated Electrons",
        "cond-mat.supr-con": "Superconductivity",
        "gr-qc": "General Relativity and Quantum Cosmology",
        "hep-ex": "High Energy Physics - Experiment",
        "hep-lat": "High Energy Physics - Lattice",
        "hep-ph": "High Energy Physics - Phenomenology",
        "hep-th": "High Energy Physics - Theory",
        "math-ph": "Mathematical Physics",
        "nlin": "Nonlinear Sciences",
        "nlin.AO": "Adaptation and Self-Organizing Systems",
        "nlin.CG": "Cellular Automata and Lattice Gases",
        "nlin.CD": "Chaotic Dynamics",
        "nlin.SI": "Exactly Solvable and Integrable Systems",
        "nlin.PS": "Pattern Formation and Solitons",
        "nucl-ex": "Nuclear Experiment",
        "nucl-th": "Nuclear Theory",
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis, Statistics and Probability",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics",
        "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Physics and Society",
        "physics.space-ph": "Space Physics",
        "quant-ph": "Quantum Physics",
    },
    "Mathematics": {
        "math.AC": "Commutative Algebra",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History and Overview",
        "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology",
        "math.LO": "Logic",
        "math.MG": "Metric Geometry",
        "math.MP": "Mathematical Physics",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings and Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
    }
}

class ArxivService(BasePaperService):
    """
    Service implementation for arXiv papers.
    """
    
    def extract_identifier(self, doi: str) -> Optional[str]:
        """
        Extract arXiv ID from a DOI string.
        
        Args:
            doi (str): A DOI string that may contain an arXiv identifier
            
        Returns:
            Optional[str]: The extracted arXiv ID if found, otherwise None
        """
        if not doi:
            return None
            
        # If DOI is already an arXiv ID
        if doi.startswith("arxiv:"):
            arxiv_id = doi.split("arxiv:")[-1]
            # Remove version if present
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]
            return arxiv_id
            
        # Check for arXiv format DOIs like 10.48550/arXiv.2106.00123
        arxiv_doi_pattern = r"10\.48550/arXiv\.(\d{4}\.\d{5})"
        match = re.search(arxiv_doi_pattern, doi)
        if match:
            logger.debug(f"Extracted arXiv ID {match.group(1)} from DOI {doi}")
            return match.group(1)
        
        # Check for direct arXiv IDs like arxiv:2106.00123v1
        direct_arxiv_pattern = r"arxiv:(\d{4}\.\d{5})(?:v\d+)?"
        match = re.search(direct_arxiv_pattern, doi, re.IGNORECASE)
        if match:
            logger.debug(f"Extracted arXiv ID {match.group(1)} from direct ID {doi}")
            return match.group(1)
        
        logger.debug(f"Could not extract arXiv ID from DOI: {doi}")
        return None
    
    def fetch_full_text(self, paper: ResearchPaper) -> str:
        """
        Fetches the full text of an arXiv paper.
        
        Args:
            paper (ResearchPaper): A ResearchPaper object with arxiv ID in the DOI
            
        Returns:
            str: The full text of the paper, or enhanced metadata if full text retrieval fails
        """
        logger.info(f"Fetching full text for arXiv paper with DOI: {paper.doi}")
        
        try:
            # Extract arXiv ID from DOI
            arxiv_id = self.extract_identifier(paper.doi)
            
            if not arxiv_id:
                logger.warning(f"Could not extract arXiv ID from DOI: {paper.doi}")
                return paper.abstract if paper.abstract else ""
            
            # First attempt: retrieve the actual PDF
            #pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
            logger.info(f"Attempting to download PDF from {pdf_url}")
            
            try:
                # Download the PDF using make_api_request instead of requests.get
                response = make_api_request(pdf_url, f"PDF for arXiv ID {arxiv_id}")
                if response and response.status_code == 200 and response.content:
                    # Extract text from the PDF using the utility function from services.utils
                    pdf_text = pdf_to_text(response.content)
                    
                    if pdf_text:
                        logger.info(f"Successfully extracted {len(pdf_text)} characters from PDF for arXiv ID: {arxiv_id}")
                        
                        # Get basic metadata
                        metadata = self.fetch_paper_metadata(arxiv_id)
                        
                        # Combine metadata and full text
                        components = []
                        
                        # Add title
                        if metadata.get("title"):
                            components.append(f"Title: {metadata.get('title')}")
                        
                        # Add authors
                        if metadata.get("authors"):
                            components.append(f"Authors: {metadata.get('authors')}")
                        
                        # Add categories
                        if metadata.get("categories"):
                            components.append(f"Categories: {metadata.get('categories')}")
                        
                        # Add full text
                        components.append(f"Full Text:\n\n{pdf_text}")
                        
                        # Combine all components
                        return "\n\n".join(components)
            
            except ImportError:
                logger.warning("PDF extraction libraries not available. Falling back to metadata.")
            except Exception as e:
                logger.warning(f"Error downloading or processing PDF: {e}. Falling back to metadata.")
            
            # Fallback: If we couldn't get the PDF, fetch enhanced metadata
            logger.info(f"Falling back to metadata for arXiv ID: {arxiv_id}")
            metadata = self.fetch_paper_metadata(arxiv_id)
            
            if not metadata:
                logger.warning(f"Could not fetch metadata for arXiv ID: {arxiv_id}")
                return paper.abstract if paper.abstract else ""
            
            # Format metadata into a comprehensive text
            components = []
            
            # Add title
            if metadata.get("title"):
                components.append(f"Title: {metadata.get('title')}")
            
            # Add abstract
            if metadata.get("abstract"):
                components.append(f"Abstract: {metadata.get('abstract')}")
            
            # Add authors
            if metadata.get("authors"):
                components.append(f"Authors: {metadata.get('authors')}")
            
            # Add categories
            if metadata.get("categories"):
                components.append(f"Categories: {metadata.get('categories')}")
            
            # Add publication date
            if metadata.get("date"):
                components.append(f"Published: {metadata.get('date')}")
            
            # Add arXiv links
            if metadata.get("pdf_link"):
                components.append(f"PDF Link: {metadata.get('pdf_link')}")
            
            if metadata.get("doi_link"):
                components.append(f"DOI Link: {metadata.get('doi_link')}")
            
            # Add arXiv ID reference
            components.append(f"arXiv ID: {arxiv_id}")
            
            # Combine all components
            enhanced_text = "\n\n".join(components)
            
            logger.info(f"Successfully compiled enhanced metadata for arXiv ID: {arxiv_id}")
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error fetching full text for arXiv paper: {e}")
            return paper.abstract if paper.abstract else ""
            
    def fetch_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Fetch metadata for an arXiv paper using its ID.
        
        Args:
            arxiv_id (str): The arXiv ID
            
        Returns:
            Dict[str, Any]: Dictionary containing paper metadata
        """
        try:
            response = self.fetch_arxiv_paper(arxiv_id)
            if not response:
                logger.warning(f"No response received for arXiv ID: {arxiv_id}")
                return {}
                
            # Parse the response
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(response.content)
            entry = root.find('.//atom:entry', namespaces)
            
            if not entry:
                logger.warning(f"No entry found for arXiv ID: {arxiv_id}")
                return {}
            
            # Extract metadata
            metadata = {}
            
            # Extract title
            title_elem = entry.find('./atom:title', namespaces)
            if title_elem is not None and title_elem.text:
                metadata["title"] = title_elem.text.strip()
            
            # Extract abstract
            summary_elem = entry.find('./atom:summary', namespaces)
            if summary_elem is not None and summary_elem.text:
                metadata["abstract"] = summary_elem.text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall('./atom:author', namespaces):
                name_elem = author.find('./atom:name', namespaces)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            
            if authors:
                metadata["authors"] = ", ".join(authors)
            
            # Extract categories
            categories = []
            for category in entry.findall('./arxiv:primary_category', namespaces):
                if category.get('term'):
                    categories.append(category.get('term'))
            
            for category in entry.findall('./atom:category', namespaces):
                if category.get('term') and category.get('term') not in categories:
                    categories.append(category.get('term'))
            
            if categories:
                metadata["categories"] = ", ".join(categories)
            
            # Extract publication date
            published_elem = entry.find('./atom:published', namespaces)
            if published_elem is not None and published_elem.text:
                metadata["date"] = published_elem.text[:10]  # Just the date part
            
            # Extract arXiv links
            for link in entry.findall('./atom:link', namespaces):
                if link.get('title') == 'pdf':
                    metadata["pdf_link"] = link.get('href')
                elif link.get('title') == 'doi':
                    metadata["doi_link"] = link.get('href')
            
            # Add the arXiv ID
            metadata["arxiv_id"] = arxiv_id
            metadata["doi"] = f"arxiv:{arxiv_id}"
            
            logger.info(f"Successfully retrieved metadata for arXiv ID: {arxiv_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for arXiv ID {arxiv_id}: {e}")
            return {}
    
    def fetch_arxiv_paper(self, arxiv_id: str):
        """
        Fetch details for a specific arXiv paper.
        
        Args:
            arxiv_id (str): The arXiv ID
            
        Returns:
            Response object or None if request failed
        """
        arxiv_api_url = 'http://export.arxiv.org/api/query'
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        encoded_params = urllib.parse.urlencode(params)
        query_url = f"{arxiv_api_url}?{encoded_params}"
        logger.info(f"Fetching single arXiv paper with ID: {arxiv_id}")
        return make_api_request(query_url, f"arXiv paper ID {arxiv_id}")
    
    def fetch_arxiv_details(self, search_query: str, start_idx: int, max_results: int):
        """
        Fetch paper details from arXiv API using a search query.
        
        Args:
            search_query (str): The search query
            start_idx (int): Starting index for pagination
            max_results (int): Maximum number of results to return
            
        Returns:
            Response object or None if request failed
        """
        arxiv_api_url = 'http://export.arxiv.org/api/query'
        params = {
            'search_query': search_query,
            'start': start_idx,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        encoded_params = urllib.parse.urlencode(params)
        query_url = f"{arxiv_api_url}?{encoded_params}"
        logger.info(f"Fetching arXiv papers with query: {search_query}, start: {start_idx}, max: {max_results}")
        return make_api_request(query_url, "arXiv details")
    
    def run_queries(self, queries: List[str], max_results: int = 10) -> List[ResearchPaper]:
        """
        Execute each query against the arXiv service and return a list of ResearchPaper objects.
        
        Args:
            queries: List of search queries to execute against arXiv
            max_results: Maximum number of results to return per query
            
        Returns:
            List of ResearchPaper objects from query results
        """
        papers = []
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        logger.info(f"Running {len(queries)} queries against arXiv")
        
        for q in queries:
            # Search arXiv with the query
            response = self.fetch_arxiv_details(search_query=q, start_idx=0, max_results=max_results)
            
            if not response:
                logger.warning(f"No response from arXiv for query: {q}")
                continue
                
            try:
                # Parse the XML response
                root = ET.fromstring(response.content)
                entries = root.findall('.//atom:entry', namespaces)
                
                logger.info(f"Query '{q}' returned {len(entries)} results from arXiv")
                
                # Process each entry into a ResearchPaper
                for entry in entries:
                    paper = self.create_arxiv_paper(entry, namespaces)
                    if paper:
                        papers.append(paper)
                        
            except ET.ParseError as e:
                logger.error(f"Error parsing arXiv response: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing arXiv results: {e}")
        
        logger.info(f"Retrieved {len(papers)} total papers from arXiv")
        return papers
    
    def create_arxiv_paper(self, entry, namespaces) -> Optional[ResearchPaper]:
        """
        Create a ResearchPaper object from arXiv API data.
        
        Args:
            entry: XML element containing arXiv paper data
            namespaces: XML namespaces
            
        Returns:
            ResearchPaper object or None if creation fails
        """
        try:
            arxiv_id = entry.find('./atom:id', namespaces).text.split('/')[-1]
            doi = f"arxiv:{arxiv_id}"
            
            title_elem = entry.find('./atom:title', namespaces)
            title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else ""
            
            abstract_elem = entry.find('./atom:summary', namespaces)
            abstract = abstract_elem.text.replace('\n', ' ').strip() if abstract_elem is not None else ""
            
            author_elements = entry.findall('./atom:author', namespaces)
            authors = "; ".join([a.find('./atom:name', namespaces).text for a in author_elements 
                               if a.find('./atom:name', namespaces) is not None])
            
            published_elem = entry.find('./atom:published', namespaces)
            date = published_elem.text[:10] if published_elem is not None else ""
            
            category_elements = entry.findall('./arxiv:primary_category', namespaces)
            category = category_elements[0].get('term') if category_elements else ""
            
            logger.debug(f"Created ResearchPaper object for arXiv ID: {arxiv_id}, title: {title[:30]}...")
            
            return ResearchPaper(
                doi=doi,
                title=title,
                authors=authors,
                date=date,
                abstract=abstract,
                category=category,
                license="",  # arXiv doesn't provide license info in the API
                version="1",  # Default to 1 since version info is not readily available
                author_corresponding="",  # Not available from arXiv API
                author_corresponding_institution="",  # Not available from arXiv API
                published_journal="",
                published_date="",
                published_doi="",
                inclusion_decision=None,
                criteria_assessment=None,
                assessment_explanation=None,
                assessment_datetime=None  # Explicitly set to None to ensure it's blank
            )
        except Exception as e:
            logger.error(f"Error creating ResearchPaper from arXiv data: {e}")
            return None


# Create a singleton instance
arxiv_service = ArxivService()

# Function for backward compatibility
def fetch_full_text(paper: ResearchPaper) -> str:
    """
    Fetches the full text of an arXiv paper.
    
    Args:
        paper (ResearchPaper): A ResearchPaper object with arxiv ID in the DOI
        
    Returns:
        str: The full text of the paper if available, or abstract if full text cannot be retrieved
    """
    return arxiv_service.fetch_full_text(paper)

def fetch_arxiv_paper(arxiv_id):
    return arxiv_service.fetch_arxiv_paper(arxiv_id)

def fetch_arxiv_details(search_query, start_idx, max_results):
    return arxiv_service.fetch_arxiv_details(search_query, start_idx, max_results)

def extract_arxiv_id_from_doi(doi):
    return arxiv_service.extract_identifier(doi)

def run_queries(queries: list[str], max_results: int = 10) -> list[ResearchPaper]:
    return arxiv_service.run_queries(queries, max_results)

def create_arxiv_paper(entry, namespaces):
    """
    Function for backward compatibility - delegates to arxiv_service.create_arxiv_paper
    """
    return arxiv_service.create_arxiv_paper(entry, namespaces)
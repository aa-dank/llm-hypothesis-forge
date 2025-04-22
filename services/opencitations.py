import httpx
import logging
from functools import lru_cache
from typing import Set, List

logger = logging.getLogger(__name__)

OPENCITATIONS_API = "https://opencitations.net/index/api/v1/citations/{}"

@lru_cache(maxsize=128)
def load_citing_dois(test_doi: str) -> Set[str]:
    """
    Fetch DOIs of papers citing `test_doi` from the OpenCitations API.
    Results are cached using lru_cache to improve performance on repeated calls.

    Args:
        test_doi: The DOI of the test paper.

    Returns:
        A set of DOIs for papers that cite the test paper, per OpenCitations.
    """
    try:
        logger.info(f"Fetching citing DOIs from OpenCitations for: {test_doi}")
        resp = httpx.get(OPENCITATIONS_API.format(test_doi), timeout=30.0)
        resp.raise_for_status()
        records = resp.json()  # each record has 'citing' and 'cited'
        citing_dois = {rec["citing"] for rec in records if "citing" in rec}
        logger.info(f"Found {len(citing_dois)} citing DOIs from OpenCitations")
        return citing_dois
    except httpx.HTTPError as e:
        logger.error(f"HTTP error from OpenCitations API: {e}")
        return set()
    except Exception as e:
        logger.error(f"Error fetching citing DOIs from OpenCitations: {e}")
        return set()
# rag_orchestration/pipeline.py
"""High‑level helper that builds a Retrieval‑Augmented‑Generation (RAG) context
for a *single* test paper **and** (optionally) stores perplexity‑score events
for each available abstract variant.
"""

from __future__ import annotations

import datetime
import json
import logging
import zlib
from typing import List, Optional, Dict, Any, Union, Tuple

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

from data.db import get_db_session, close_db_session
from data.models import ResearchPaper, PerplexityScoreEvent
from prompts_library.rag import service_query_creation_template
from rag_orchestration.models import *
from rag_orchestration.utils import (
    cull_citing_papers,
    dispatch_queries,
    generate_service_queries,
    get_or_fetch_research_paper_text,
    mask_abstract_differences,
    replace_bracketed_content,
    generate_rag_context,
    assemble_test_abstract_prompt,
)

logger = logging.getLogger(__name__)


def construct_rag_context_for_research_paper(
    paper: ResearchPaper,
    llm_client: "BasicOpenAI | None" = None,
    *,
    generated_abstract: str | None = None,
    rag_size_chars: int = 15_000,
    avg_chunk_chars: int = 1_000,
    top_k_chunks: int | None = None,
    return_metadata: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """Return a RAG context ready for perplexity scoring.

    Parameters
    ----------
    paper : ResearchPaper
        The *test* paper.
    llm_client : BasicOpenAI | None, optional
        Used only to draft the multi‑service query plan.
    generated_abstract : str | None, optional
        If you already created a bracket‑masked abstract, pass it here.
    rag_size_chars : int, default 15 000
        Maximum number of context characters (excluding the abstract itself).
    avg_chunk_chars : int, default 1 000
        Heuristic for deriving *k* when *top_k_chunks* isn't supplied.
    top_k_chunks : int | None
        Override the automatic *k* calculation.
    return_metadata : bool, default False
        If True, return a tuple of (context_string, metadata_dict) instead of just the context string.
    """
    metadata = {
        "original_chars": 0,
        "truncated_chars": 0,
        "candidate_papers": 0,
        "culled_papers": 0,
        "chunks_retrieved": 0,
        "used_fallback": False
    }

    try:
        # 1. Mask the abstract
        logger.info("RAG build – paper‑id=%s DOI=%s", paper.id, paper.doi)
        if generated_abstract and "[[" in generated_abstract:
            masked_abstract = replace_bracketed_content(generated_abstract)
            logger.info("Using caller‑supplied pre‑masked abstract")
        elif getattr(paper, "gpt4_incorrect_abstract", None):
            masked_abstract = mask_abstract_differences(
                abstract_a=paper.abstract,
                abstract_b=paper.gpt4_incorrect_abstract,
            )
            logger.info("Masked real × GPT‑4 abstract differences")
        else:
            raise ValueError("Need either *generated_abstract* or *paper.gpt4_incorrect_abstract* for masking")

        # 2. Generate service query plan via LLM
        from services.llm_services import BasicOpenAI  # lazy heavy‑dep import
        if llm_client is None:
            llm_client = BasicOpenAI()

        query_plan = generate_service_queries(
            masked_abstract=masked_abstract,
            llm_client=llm_client,
            template_str=service_query_creation_template,
        )
        if not query_plan:
            logger.error("Query plan empty – aborting")
            if return_metadata:
                return "", metadata
            return ""

        # 3. Dispatch queries to external services
        results = dispatch_queries(query_plan, max_results_per_query=5)
        if not results:
            logger.warning("All services returned zero results – aborting")
            if return_metadata:
                return "", metadata
            return ""

        candidate_papers: List[ResearchPaper] = [p for papers in results.values() for p in papers]
        logger.info("Candidates before culling: %d", len(candidate_papers))
        metadata["candidate_papers"] = len(candidate_papers)

        # 4. Cull citing papers (leak‑proofing)
        culled_papers = cull_citing_papers(candidate_papers, test_doi=paper.doi, include_opencitations=True)
        if not culled_papers:
            logger.warning("Every candidate cites the test paper – aborting")
            if return_metadata:
                return "", metadata
            return ""
        logger.info("After culling: %d papers", len(culled_papers))
        metadata["culled_papers"] = len(culled_papers)

        # 5. Ensure full‑text chunks + embeddings are stored
        session: Session = get_db_session()
        try:
            for rp in culled_papers:
                try:
                    _ = get_or_fetch_research_paper_text(rp, session)
                except Exception:
                    logger.exception("Could not fetch/ingest text for DOI %s", rp.doi)

            # 6. Vector similarity search → top‑k chunks
            k = top_k_chunks or max(5, min(30, rag_size_chars // avg_chunk_chars))
            logger.info("Vector search with k=%d (rag_size_chars=%d)", k, rag_size_chars)
            rag_text, chunks_retrieved = generate_rag_context(session, masked_abstract, k=k, return_chunk_count=True)
            metadata["chunks_retrieved"] = chunks_retrieved
            
            if not rag_text:
                logger.warning("Vector search empty – concatenating first 10 full texts as fallback")
                rag_text = "\n\n---\n\n".join(
                    (get_or_fetch_research_paper_text(p, session) or "") for p in culled_papers[:10]
                )
                metadata["used_fallback"] = True
        finally:
            close_db_session(session)

        # 7. Process final context (truncate if needed)
        metadata["original_chars"] = len(rag_text)
        if len(rag_text) > rag_size_chars:
            logger.info("Truncating RAG text from %d → %d chars", len(rag_text), rag_size_chars)
            rag_text = rag_text[:rag_size_chars]
            # Snap to a clean break if we can
            for sep in (". ", ".\n", "\n\n", "\n"):
                pos = rag_text.rfind(sep, max(0, rag_size_chars - 200))
                if pos != -1:
                    rag_text = rag_text[: pos + len(sep)]
                    break
        metadata["truncated_chars"] = len(rag_text)

        # Return just the RAG context without appending the test abstract
        final_context = rag_text.strip()
        logger.info("Final RAG context size = %d chars", len(final_context))
        
        if return_metadata:
            return final_context, metadata
        return final_context

    except Exception as exc:
        logger.error("RAG construction failed for paper %s: %s", paper.id, exc)
        import traceback
        logger.error(traceback.format_exc())
        if return_metadata:
            return "", metadata
        return ""

def score_rag_perplexity_for_paper(
    paper: ResearchPaper,
    *,
    together_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    rag_size_chars: int = 15_000,
    session: Session | None = None,
) -> List[PerplexityScoreEvent]:
    """Build a RAG prompt for *paper* for each abstract variant (original + GPT‑4)
    and store :class:`PerplexityScoreEvent` rows.

    Returns
    -------
    list[PerplexityScoreEvent]
        The newly‑created (but *committed*) events.
    """

    from services.llm_services import TogetherClient  # local import to avoid heavy deps

    own_session = False
    if session is None:
        session = get_db_session()
        own_session = True

    events: List[PerplexityScoreEvent] = []
    try:
        tc = TogetherClient(model=together_model)

        # Helper to add an event
        def _insert_event(source: str, abstract_text: str, generated_abstract: str | None):
            
            rag_context, metadata = construct_rag_context_for_research_paper(
                paper,
                llm_client=None,  # fresh BasicOpenAI inside
                generated_abstract=generated_abstract,
                rag_size_chars=rag_size_chars,
                return_metadata=True,
            )
            if not rag_context:
                logger.warning("Empty RAG context for paper %s source=%s – skipping", paper.id, source)
                return

            # Use the utility function to assemble the full prompt
            full_prompt = assemble_test_abstract_prompt(rag_context.strip(), abstract_text.strip())
            logger.info(f"Final prompt size with abstract ({source}): {len(full_prompt)} chars")

            perplexity = tc.perplexity_score(full_prompt)
            zlib_size = len(zlib.compress(full_prompt.encode("utf-8")))
            zlib_ratio = perplexity / zlib_size if zlib_size else None

            meta: Dict[str, Any] = {
                "method": "rag",
                "rag_size_chars": rag_size_chars,
                "abstract_source": source,
                **metadata,
            }

            evt = PerplexityScoreEvent(
                research_paper_id=paper.id,
                model=together_model,
                abstract_text=abstract_text,
                abstract_source=source,
                prompt_template_name="rag_context",  # not really a template but fits schema
                full_prompt=full_prompt,
                zlib_compression_size=zlib_size,
                perplexity_score=perplexity,
                zlib_perplexity_ratio=zlib_ratio,
                evaluation_datetime=datetime.datetime.utcnow(),
                additional_metadata=meta,
            )
            session.add(evt)
            events.append(evt)
            session.commit()
        # --- Original abstract
        if paper.abstract:
            _insert_event("original", paper.abstract, None)

        # --- GPT‑4 incorrect abstract
        if getattr(paper, "gpt4_incorrect_abstract", None):
            _insert_event("gpt4", paper.gpt4_incorrect_abstract, paper.gpt4_incorrect_abstract)
            
        # --- Human incorrect abstract (if available)
        if getattr(paper, "human_incorrect_abstract", None):
            _insert_event("human", paper.human_incorrect_abstract, paper.human_incorrect_abstract)

        session.commit()
        return events

    except IntegrityError:
        logger.exception("IntegrityError while inserting perplexity events for paper %s", paper.id)
        session.rollback()
        return events

    finally:
        if own_session:
            close_db_session(session)

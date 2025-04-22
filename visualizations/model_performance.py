import json
from sqlalchemy import func
from sqlalchemy.orm import aliased
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.db import get_db_session
from data.models import PerplexityScoreEvent


def get_min_perplexity_rows(session):
    min_subq = (
        session.query(
            PerplexityScoreEvent.research_paper_id,
            func.min(PerplexityScoreEvent.perplexity_score).label("min_score")
        )
        .group_by(PerplexityScoreEvent.research_paper_id)
        .subquery()
    )

    ps_alias = aliased(PerplexityScoreEvent)

    return (
        session.query(ps_alias)
        .join(
            min_subq,
            (ps_alias.research_paper_id == min_subq.c.research_paper_id) &
            (ps_alias.perplexity_score == min_subq.c.min_score)
        )
        .with_entities(
            ps_alias.research_paper_id,
            ps_alias.abstract_source
        )
        .all()
    )


def count_sources(rows):
    counts = {"True": 0, "False": 0}
    for row in rows:
        source = row.abstract_source.lower()

        if source == "original":
            counts["True"] += 1
        elif source in {"gpt4", "human"}:
            counts["False"] += 1

    return counts


def save_counts_to_file(counts, filename="perplexity_source_counts.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    with open(file_path, "w") as f:
        json.dump(counts, f, indent=2)


def main():
    session = get_db_session()
    rows = get_min_perplexity_rows(session)
    counts = count_sources(rows)
    save_counts_to_file(counts)
    print("Counts saved to perplexity_source_counts.json")


if __name__ == "__main__":
    main()

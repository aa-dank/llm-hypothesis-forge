import re
import random
import pandas as pd
import json
from tqdm import tqdm
from sqlalchemy import text
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.utils import get_db_session
from prompts_library.data_creation import discern_abstracts_pair_prompt
from services.llm_services import BasicOpenAI

pattern = re.compile(r"\[\[\s*['\"]?(.*?)['\"]?\s*,\s*['\"]?(.*?)['\"]?\s*\]\]")


def replace_double_brackets(text, which='second'):
    if not isinstance(text, str):
        return ""
    return pattern.sub(lambda m: m.group(1) if which == 'first' else m.group(2), text)


def fetch_abstract_pairs(session):
    query = text("""
        SELECT abstract, 
               COALESCE(NULLIF(human_incorrect_abstract, ''), NULLIF(gpt4_incorrect_abstract, '')) AS generated_abstract
        FROM research_papers
        WHERE abstract IS NOT NULL
          AND (human_incorrect_abstract IS NOT NULL OR gpt4_incorrect_abstract IS NOT NULL)
    """)
    return session.execute(query).fetchall()


def format_prompt_and_labels(orig, gen):
    if random.random() < 0.5:
        return (orig, gen, 1)  # real abstract is abstract_1
    else:
        return (gen, orig, 2)  # real abstract is abstract_2


def clean_llm_response(response):
    response_cleaned = response.lstrip("```json").rstrip("```")
    return response_cleaned.replace(r'\\', r'\\\\')


def parse_llm_response(response):
    try:
        cleaned = clean_llm_response(response)
        response_json = json.loads(cleaned)
        return int(response_json['decision'])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response:\n{response}\n{e}")
        return None


def evaluate_abstracts(llm_client, abstracts):
    true_decisions = false_decisions = 0

    for row in tqdm(abstracts, desc="Processing abstracts"):
        orig = replace_double_brackets(row.abstract, which='first')
        gen = replace_double_brackets(row.generated_abstract, which='second')
        abstract_1, abstract_2, answer_key = format_prompt_and_labels(orig, gen)

        prompt = discern_abstracts_pair_prompt.format(
            abstract_1=abstract_1,
            abstract_2=abstract_2
        )

        response = llm_client.complete(
            prompt=prompt,
            system_message="You are a helpful research assistant.",
            temperature=0.7,
            max_tokens=None
        )

        decision = parse_llm_response(response)
        if decision is None:
            continue

        if decision == answer_key:
            true_decisions += 1
        else:
            false_decisions += 1

    return pd.DataFrame([{
        'True Decisions': true_decisions,
        'False Decisions': false_decisions
    }])

def main():
    session = get_db_session()
    llm_client = BasicOpenAI()
    abstracts = fetch_abstract_pairs(session)
    stats_df = evaluate_abstracts(llm_client, abstracts)
    
    # Save CSV file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'discern_abstracts_stats.csv')
    
    stats_df.to_csv(file_path, index=False)
    print(f"Stats saved to {file_path}")

if __name__ == "__main__":
    main()

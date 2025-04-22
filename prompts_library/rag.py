#prompts_library/rag.py

service_query_creation_template = """
You are an expert research assistant helping to create precise search queries for scientific document retrieval.

I have a research abstract with masked information (marked as [[DIFF]]). Your task is to create appropriate search queries 
for various scientific databases to help retrieve papers that might contain the missing information.

Abstract with masked sections:
""""""  
{{masked_abstract}}
""""""  

For each service listed below, generate 2-3 search queries that would best retrieve information to fill in the masked sections.
Consider the domain expertise and specific strengths of each database.

Available services:
{% for service in services %}
- {{service.name}}: {{service.description}}
  Query format: {{service.query_format}}
{% endfor %}

Respond with a JSON object with the following structure:
```
{
  "queries": [
    {
      "service": "ServiceName",
      "queries": ["query1", "query2", "query3"]
    },
    {...}
  ]
}
```

Be precise and consider what specific keywords or combinations would retrieve the most relevant papers.
"""

rag_context_template = """
You are an expert research assistant. Here is relevant information about a research paper:
{{rag_context}}


Here is an abstract from a related paper:
{{abstract}}
"""
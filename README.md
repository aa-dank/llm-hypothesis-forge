# LLM Research Outcome Prediction: Cross-Disciplinary Benchmarking

## Overview

This project investigates the capability of Large Language Models (LLMs) to perform scientific research evaluation across disciplines through the development of a comprehensive benchmarking framework. Inspired by the groundbreaking paper "Large Language Models Surpass Human Experts in Predicting Neuroscience Results" (Luo et al., 2024), we expand the abstract categorization benchmarking methodology to evaluate LLM performance in distinguishing between authentic and fabricated research abstracts across multiple domains. Also this project serves as a prototype for a production benchmarking system that can be utilized by organizations to assess the effectiveness of LLM tools in research applications.

## Motivation

In today's rapidly evolving research landscape, scientists and organizations need powerful tools to efficiently process vast amounts of literature. LLMs have shown promise in this area, with applications ranging from literature review automation to hypothesis generation. However, the effectiveness of these models can vary significantly depending on the specific task and domain. For instance, while LLMs may excel at generating coherent text, their ability to accurately predict research outcomes or evaluate methodologies is still under scrutiny.
But how do organizations assess which LLM tools are most effective for these specialized tasks? Our project addresses this crucial need by developing a robust benchmarking system for evaluating LLM performance in scientific research evaluation.

## Project Goals

1. **Extend Abstract Categorization Benchmarking**: Building on Luo et al.'s methodology, we extend the evaluation framework beyond neuroscience to include macroeconomics and optometry.

2. **Test State-of-the-Art Models**: We evaluate modern, advanced LLMs against established benchmarks to understand their comparative performance.

3. **Explore RAG Architecture Benefits**: We investigate whether Retrieval-Augmented Generation (RAG) approaches enhance LLM performance in research evaluation tasks.

4. **Develop Production Benchmarking System**: We aim to create a prototype benchmarking system that could be used by organizations to evaluate LLM tools for research applications.

## Methodology

Our approach involves several key phases:

1. **Data Collection**: We source research paper data from multiple databases including Arxiv, Bioarxiv, Elsevier, Pubmed, and OpenAlex, focusing on three domains:
   - Neuroscience (maintaining comparability with Brainbench)
   - Optometry (chosen for its highly technical language)
   - Macroeconomics (for disciplinary diversity)

2. **Data Processing**: Papers undergo domain-specific filtering for relevance and quality, with GPT-4o providing structured assessments.

3. **LLM Selection**: We test multiple model classes including OpenAI's GPT, Google Gemini, and Deepseek.

4. **Evaluation Framework**: We generate convincing but factually incorrect abstract variants and evaluate model performance using perplexity scores, accuracy metrics, and confidence analysis.

5. **Infrastructure**: We deploy a PostgreSQL database hosted on AWS to store research paper metadata, citations relationships, and vector embeddings.

## Potential Impact

Our research aims to enhance decision-making in scientific fields by improving the evaluation and selection of LLM tools for research applications. This could accelerate innovation by reducing redundant experiments and helping researchers more efficiently synthesize information across vast literature landscapes.

## Repository Structure

Our repository is organized into several key modules, each handling specific aspects of the benchmarking system:

### `/data`
- Database connection and management (`db.py`)
- Data ingestion pipeline (`ingest.py`)
- Data generation utilities for creating synthetic abstracts (`generate.py`)
- Data models and schemas (`models.py`)

### `/services`
- Integration with research databases:
  - `arxiv.py`: ArXiv API client for computer science papers
  - `biorxiv.py`: BioRxiv client for preprint biological research
  - `elsevier.py`: Elsevier API integration for accessing published research
  - `pubmed.py`: PubMed integration for medical research papers
  - `openalex.py`: OpenAlex client for open research metadata
  - `unpaywall.py`: Access to open-access versions of research papers
- LLM provider integrations (`llm_services.py`)
- Base service model for API interactions (`service_model.py`)

### `/rag_orchestration`
- RAG pipeline implementation (`pipeline.py`)
- RAG-specific data models (`models.py`)
- Utility functions for RAG operations (`utils.py`)

### `/prompts_library`
- Templated prompts for data creation (`data_creation.py`)
- RAG-specific prompts (`rag.py`)

### `/visualizations`
- Visualization generation scripts (`generate_visuals.py`)
- Model performance analysis (`model_performance.py`, `gpt_evaluation.py`)
- Pre-generated visualization images in `/images`

### `/Notebooks`
- Interactive notebooks for demonstrations and experimentation

## Features

Our project includes a comprehensive set of features designed for benchmarking LLM performance in scientific research evaluation:

### Data Collection & Processing
- **Multi-source Research Paper Extraction**: Automated extraction from Arxiv, BioRxiv, Elsevier, PubMed, and OpenAlex
- **Domain-specific Filtering**: Custom filters for neuroscience, optometry, and macroeconomics papers
- **Metadata Enrichment**: Citation network mapping and paper relevance assessment
- **Vector Embeddings**: Semantic embedding generation for content retrieval

### LLM Benchmarking
- **Model-agnostic Testing Framework**: Compatible with OpenAI, Google Gemini, and Deepseek models
- **Abstract Authenticity Classification**: Tests models' ability to distinguish between real and generated abstracts
- **Perplexity Scoring**: Quantitative measurement of model confidence and accuracy
- **Cross-disciplinary Evaluation**: Comparative analysis across different scientific domains

### RAG Implementation
- **Knowledge Retrieval**: Context-aware document retrieval system
- **Vector Search**: Semantic similarity searching for relevant research context
- **Enhanced Reasoning**: Tests whether additional context improves model performance

### Analysis & Visualization
- **Performance Metrics Dashboard**: Comprehensive visualizations of model performance
- **Confidence Analysis**: Assessment of model calibration and confidence
- **Domain Comparison**: Visual comparison of performance across scientific domains
- **RAG vs. Base Model Analysis**: Comparative visualization of RAG's impact on performance

### Infrastructure
- **PostgreSQL Database**: Robust storage solution with vector search capabilities
- **Modular Architecture**: Easily extensible system for adding new models or data sources
- **Reproducible Environment**: Containerized setup for consistent testing environments

### Installation
Clone the repository:

```bash
git clone https://github.com/aa-dank/llm-hypothesis-forge.git
cd llm-hypothesis-forge
```

Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```


Download dependencies:
```bash
pip install -r requirements.txt
```


### Development Setup
- Set up the database connection: Update your database connection settings in an environment variables file.
- Set up the research paper extraction: Update your API keys/secrets in an environment variables files.

### API Keys
The project requires several API keys to access various research databases and LLM services. Create a `.env` file in the root directory with the following variables:

```
# Database Configuration
DB_USERNAME = "your_username"
DB_PASSWORD = "your_password"
DB_NAME = "your_database"
DB_HOST = "your_host"
DB_PORT = "5432"
DATABASE_TYPE = "postgresql"

# LLM API Keys
OPENAI_API_KEY = "your_openai_api_key"
GOOGLE_API_KEY = "your_google_api_key"
HUGGINGFACE_API_KEY = "your_huggingface_api_key"
TOGETHER_API_KEY = "your_together_api_key"

# Research Database API Keys
ELSEVIER_API_KEY = "your_elsevier_api_key"
ELSEVIER_INSTTOKEN = "your_elsevier_insttoken"
UNPAYWALL_EMAIL = "your_email@example.com"
```

To obtain the API keys:
- **OpenAI**: Register at [platform.openai.com](https://platform.openai.com)
- **Google**: Get API keys from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HuggingFace**: Create an account at [huggingface.co](https://huggingface.co) and generate an API key
- **TogetherAI**: Register at [together.ai](https://www.together.ai)
- **Elsevier**: Apply for API access at [dev.elsevier.com](https://dev.elsevier.com)

## Project Services & Infrastructure

| Service Category | Service Name | Description | Role in Project |
|:-----------------|:-------------|:------------|:----------------|
| **LLM Providers** | **OpenAI** | Provider of state-of-the-art language models including GPT-4o and text-embedding models that excel at natural language understanding and generation tasks. | Powers our core hypothesis generation system by analyzing research methodologies and generating outcome predictions with high accuracy. Provides embeddings for semantic search functionality through the text-embedding-3-small model. |
| | **Google Gemini** | Google's flagship generative AI offering with strong scientific reasoning capabilities and domain-specific knowledge across academic disciplines. | Offers an alternative hypothesis generation pipeline with different reasoning patterns, allowing for comparative analysis against other models. Particularly strong at interpreting complex methodological descriptions. |
| | **HuggingFace** | Open platform hosting thousands of machine learning models with specialized capabilities across numerous domains and languages. | Provides access to specialized scientific and domain-specific models that supplement our core reasoning engines with targeted knowledge. Enables experimentation with open-source alternatives. |
| | **TogetherAI** | Cloud platform that hosts and serves various open models at scale with optimized inference speeds and consistent APIs. | Allows efficient testing with models like Mistral, Llama, and DeepSeek for comparison across different model architectures and training approaches. Provides standardized log probability outputs for perplexity calculations. |
| **Research Databases** | **Elsevier** | One of the world's largest academic publishers with a comprehensive API offering access to millions of peer-reviewed articles. | Extracts high-quality research papers from prestigious journals with structured metadata, allowing our system to analyze established research methodologies and outcomes. |
| | **arXiv** | Open-access repository of electronic preprints for physics, mathematics, computer science, and related fields. | Provides access to cutting-edge research papers before formal publication, giving our system visibility into emerging methods and experimental designs. |
| | **PubMed** | Comprehensive database of biomedical literature maintained by the National Library of Medicine. | Supplies our system with high-quality medical research papers focusing on clinical trials and biomedical experimentation with validated outcomes. |
| | **bioRxiv** | Preprint server for biology research that allows researchers to make their findings immediately available. | Offers access to the latest biological research papers before peer review, extending our dataset with emerging experimental approaches. |
| | **OpenAlex** | Open catalog of scholarly papers, authors, and institutions that aims to democratize access to scholarly metadata. | Provides supplementary paper information and citation networks to enhance our understanding of research impact and relationships. |
| | **Unpaywall** | Database of open-access scholarly articles that identifies legally free versions of paywalled research. | Helps our system access full-text content that might otherwise be behind paywalls, significantly increasing our training data coverage. |
| **Data Infrastructure** | **PostgreSQL** | Robust, open-source relational database management system with native vector storage extensions. | Stores and manages our collection of research papers, methodologies, LLM predictions, and vector embeddings with high reliability and performance. |
| | **pgvector** | Vector similarity search extension for PostgreSQL enabling efficient similarity queries. | Powers our semantic search functionality by enabling fast similarity searches across millions of embedded text chunks. |

Each service integration plays a crucial role in our system's ability to:
1. Extract and process relevant scientific papers from diverse academic databases
2. Transform unstructured research text into structured, queryable information
3. Generate informed hypotheses about expected experimental outcomes using various LLM architectures
4. Compare prediction accuracy across different models and scientific domains
5. Evaluate the scientific reasoning capabilities of different language models through direct comparison

## Data Access Statement
Due to data privacy agreements, the data used in this project cannot be published. The code in this repository can be used, but the database must be populated with your own data.


Contact:
aadank@umich.edu
ggmathew@umich.edu
mgtran@umich.edu

---

## Acknowledgments

This project utilizes generative AI tools to assist with code development. Python files and functions throughout this repository were developed with the help of AI coding assistants unless explicitly noted otherwise. These tools were used to accelerate development while maintaining quality standards and adhering to best practices.

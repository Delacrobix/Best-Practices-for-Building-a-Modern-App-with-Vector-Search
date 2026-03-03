# %% [markdown]
# # Best Practices for Building a Modern App with Vector Search
# 
# This notebook demonstrates how to build a modern LLM-powered application using:
# 
# - **Jina Embeddings v5** via Elastic Inference Service (EIS) GPU-accelerated multilingual embeddings
# - **Elasticsearch 9.3+** for vector storage and semantic search
# - **Agent Builder** for creating AI agents that can query your data
# 
# ## What You'll Learn
# 
# 1. Setting up inference endpoints for embeddings
# 2. Creating optimized indices for vector search
# 3. Ingesting data with automatic embedding generation
# 4. Performing semantic searches
# 5. Building an AI agent with Agent Builder

# %% [markdown]
# ## Setup and Configuration

# %%
# Install required packages
!pip install elasticsearch requests dotenv -q

# %%
import os
import json

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

# Elasticsearch configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTIC_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
KIBANA_URL = os.getenv("KIBANA_URL")

# Initialize Elasticsearch client
es_client = Elasticsearch(ELASTICSEARCH_URL, api_key=ELASTIC_API_KEY)

# %% [markdown]
# ## Tip 1: Use Managed Inference Instead of External Embedding APIs
#
# We'll use **Jina Embeddings v5** through Elastic Inference Service (EIS).
#
# EIS eliminates the need to manage ML infrastructure while providing GPU-accelerated performance.

# %%
INFERENCE_ENDPOINT_ID = ".jina-embeddings-v5-text-small"

# Jina Embeddings v5 comes with a preconfigured endpoint in EIS.
# No need to create a custom inference endpoint; just reference the preconfigured ID.
print(f"Using preconfigured inference endpoint: {INFERENCE_ENDPOINT_ID}")

# %% [markdown]
# ## Tip 2: Design Hybrid-Ready Indices from Day One
#
# Use `copy_to` at mapping time. Keep the original text fields for BM25, and automatically
# copy their content into a dedicated `semantic_text` field for vector search.

# %%
INDEX_NAME = "tech-articles"

# Create index with semantic_text field using copy_to pattern
# Best Practice: Keep original fields for BM25 search and use copy_to for semantic search
index_mappings = {
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "copy_to": "semantic_field",
                "fields": {"keyword": {"type": "keyword"}},
            },
            "content": {"type": "text", "copy_to": "semantic_field"},
            "category": {"type": "keyword"},
            "published_date": {"type": "date"},
            "semantic_field": {
                "type": "semantic_text",
                "inference_id": INFERENCE_ENDPOINT_ID,
            },
        }
    }
}

# Only create the index if it doesn't exist
if not es_client.indices.exists(index=INDEX_NAME):
    response = es_client.indices.create(index=INDEX_NAME, body=index_mappings)
    print(f"Created index: {INDEX_NAME}")
    print(json.dumps(response.body, indent=2))
else:
    print(f"Index '{INDEX_NAME}' already exists, skipping creation")

# %% [markdown]
# ## Tip 3: Use Bulk Operations for Scalable Ingestion

# %%
from elasticsearch import helpers


def build_data(json_file, index_name):
    """Generator function to yield documents for bulk indexing."""
    with open(json_file, "r") as f:
        data = json.load(f)

    for doc in data:
        yield {"_index": index_name, "_source": doc}


# Bulk index the documents from JSON file
try:
    success, failed = helpers.bulk(
        es_client,
        build_data("dataset.json", INDEX_NAME),
    )
    print(f"{success} documents indexed successfully")

    if failed:
        print(f"Errors: {failed}")
except Exception as e:
    print(f"Error: {str(e)}")

# %%
# Check document count
count = es_client.count(index=INDEX_NAME)
print(f"Total documents in index: {count.body['count']}")

# %% [markdown]
# ## Tip 4: Use a Hybrid Search Strategy
#
# Hybrid search combines BM25 (lexical) and semantic retrieval to get the benefits of both:
# the precision of term matching and the contextual understanding of dense vector search.
# With your index already designed to support both BM25 and semantic search, the question
# is how to merge the two ranked result lists.
#
# Elasticsearch provides built-in methods to combine semantic and full-text strategies
# through its retrievers framework. Here we cover **Reciprocal Rank Fusion (RRF)** and
# **linear combination** — the two methods designed to fuse semantic and BM25 results.

# %% [markdown]
# ### Reciprocal Rank Fusion (RRF)
#
# This is the starting point recommended by Elastic. It ignores raw scores and merges
# results based on rank positions alone. A document ranked 3rd in semantic and 5th in BM25
# gets a combined score based on those positions, not the raw numbers. This makes it robust
# against mismatched score scales, requires almost no tuning, and works well out of the box.
#
# The tradeoff is that RRF discards score magnitude: a document that scores dramatically
# higher than others in one retriever won't get extra credit for that gap.
#
# Elasticsearch also supports weighted RRF for scenarios where you need to boost one
# retriever over another without switching to score-based merging.

# %%
def hybrid_search_rrf(query: str, size: int = 3):
    """Hybrid search using RRF."""
    response = es_client.search(
        index=INDEX_NAME,
        body={
            "retriever": {
                "rrf": {
                    "retrievers": [
                        {
                            "standard": {
                                "query": {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["title^2", "content"],
                                    }
                                }
                            }
                        },
                        {
                            "standard": {
                                "query": {
                                    "match": {"semantic_field": {"query": query}}
                                }
                            }
                        },
                    ],
                    "rank_window_size": 50,
                    "rank_constant": 20,
                }
            },
            "size": size,
            "_source": ["title", "category"],
        },
    )

    return response

# %%
hybrid_search_rrf("What are the best practices for semantic search in Elasticsearch?")

# %% [markdown]
# ### Linear Retriever
#
# This approach takes the opposite direction: it directly merges actual scores from each
# retriever using a weighted sum. Because BM25 scores are unbounded while dense vector
# scores typically fall within [0, 1], linear combination requires normalization to bring
# scores into a comparable range.
#
# Once normalized, you control each retriever's influence through weights: a weight greater
# than 1 boosts that retriever's contribution, less than 1 reduces it. This preserves score
# differences and gives you fine-grained control, making it better suited when retrievers
# return disjoint results or when you need to balance lexical and semantic results.

# %%
def hybrid_search_linear(query: str, size: int = 3):
    """Hybrid search using linear retriever with MinMax normalization."""
    response = es_client.search(
        index=INDEX_NAME,
        body={
            "retriever": {
                "linear": {
                    "retrievers": [
                        {
                            "retriever": {
                                "standard": {
                                    "query": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["title^2", "content"],
                                        }
                                    }
                                }
                            },
                            "weight": 1.5,
                        },
                        {
                            "retriever": {
                                "standard": {
                                    "query": {
                                        "match": {"semantic_field": {"query": query}}
                                    }
                                }
                            },
                            "weight": 5,
                        },
                    ],
                    "normalizer": "minmax",
                }
            },
            "size": size,
            "_source": ["title", "category"],
        },
    )

    return response

# %%
hybrid_search_linear("What are the best practices for semantic search in Elasticsearch?")

# %% [markdown]
# RRF provides fast, reliable hybridization out of the box, while linear combination offers
# higher potential accuracy once weights and normalizers are tuned to your application and
# data. There is no universally better option — the right choice depends on your use case.

# %% [markdown]
# ## Tip 5: Add AI Reasoning with Agent Builder
#
# Elastic Agent Builder lets you create LLM-powered agents that can query your
# Elasticsearch data using natural language. The agent uses the Elastic Managed LLM
# by default, which is available on Elastic Cloud with no additional API key or
# connector configuration.

# %%
headers = {
    "kbn-xsrf": "true",
    "Authorization": f"ApiKey {ELASTIC_API_KEY}",
    "Content-Type": "application/json",
}

AGENT_ID = "tech-articles-assistant"

agent_payload = {
    "id": AGENT_ID,
    "name": "Tech Articles Assistant",
    "description": "An AI assistant that helps users find information about technology topics from our knowledge base.",
    "configuration": {
        # Uses Elastic Managed LLM by default — no connector_id needed
        "tools": [{"tool_ids": ["platform.core.search", "platform.core.execute_esql"]}],
        "instructions": f"""You are a helpful assistant that answers questions about technology topics.

Use the search tool to find relevant articles from the '{INDEX_NAME}' index.
When searching, prefer semantic search for natural language questions.
Always cite the article titles when providing information.
If you cannot find relevant information, say so clearly.""",
    },
}

response = requests.post(
    f"{KIBANA_URL}/api/agent_builder/agents",
    headers=headers,
    json=agent_payload,
    verify=True,
)

if response.status_code == 200:
    agent_data = response.json()
    print(json.dumps(agent_data, indent=2))
else:
    print(f"Error creating agent: {response.text}")
    agent_id = None

# %%
# Chat with the agent
def chat_with_agent(agent_id: str, message: str):
    """Send a message to the agent and get a response."""
    chat_payload = {"input": message, "agent_id": AGENT_ID}

    response = requests.post(
        f"{KIBANA_URL}/api/agent_builder/converse",
        headers=headers,
        json=chat_payload,
        verify=True,
    )

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text, "status_code": response.status_code}

# %%
# Example conversation
result = chat_with_agent(
    AGENT_ID, "What are the best practices for building RAG applications?"
)
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Tip 6: Automate with Elastic Workflows
#
# **Elastic Workflows** is an automation engine built into Elasticsearch that orchestrates multi-step processes using YAML.
#
# > **Note:** Workflows is in technical preview as of Elasticsearch 9.3. To enable the Workflows UI,
# > go to Kibana: `Stack Management > Advanced Settings`, and set `workflows:ui:enabled` to `true`.
# > Paste the YAML below into the editor to create this workflow.

# %%
"""
name: Hacker News Digest
description: >
  Fetches the latest top stories from the Hacker News public API, indexes them
  into Elasticsearch with semantic embeddings, then asks the AI agent to
  summarize the key themes from the freshly ingested content.
enabled: true
tags: ["ingestion", "hacker-news", "agent", "demo"]

consts:
  indexName: tech-articles
  hnApiBase: "https://hacker-news.firebaseio.com/v0"
  agentId: tech-articles-assistant

triggers:
  - type: manual

steps:
  # Step 1: Fetch top story IDs from the Hacker News public API (no auth required)
  - name: fetch_top_stories
    type: http
    with:
      url: "{{ consts.hnApiBase }}/topstories.json"
      method: GET

  # Step 2: For each story ID, fetch details and index into Elasticsearch.
  - name: process_stories
    type: foreach
    foreach: "${{ steps.fetch_top_stories.output.data | slice: 0, 5 }}"
    steps:
      - name: fetch_story_detail
        type: http
        with:
          url: "{{ consts.hnApiBase }}/item/{{ foreach.item }}.json"
          method: GET

      - name: index_story
        type: elasticsearch.request
        with:
          method: POST
          path: "/{{ consts.indexName }}/_doc"
          body:
            title: "{{ steps.fetch_story_detail.output.data.title }}"
            content: "{{ steps.fetch_story_detail.output.data.text | default: steps.fetch_story_detail.output.data.title }}"
            category: "hacker-news"
            url: "{{ steps.fetch_story_detail.output.data.url }}"

  # Step 3: Ask the agent to summarize the freshly indexed stories.
  - name: ask_agent
    type: ai.agent
    with:
      agent_id: "{{ consts.agentId }}"
      message: "What are the main themes and topics from the latest Hacker News stories?"

  - name: log_summary
    type: console
    with:
      message: "{{ steps.ask_agent.output }}"
"""

# %% [markdown]
# ### Results of Workflow Creation
#
# ![image.png](assets/image.png)

# %%
es_client.indices.delete(index=INDEX_NAME)



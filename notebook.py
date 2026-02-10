# -*- coding: utf-8 -*-
# %% [markdown]
# # Best Practices for Building a Modern App with Vector Search
#
# This notebook demonstrates how to build a modern LLM-powered application using:
#
# - **Jina Embeddings v3** via Elastic Inference Service (EIS) - GPU-accelerated multilingual embeddings
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
# !pip install elasticsearch requests dotenv -q

import json

# %%
import os

import requests
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

# Elasticsearch configuration
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTIC_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")
KIBANA_URL = os.getenv("KIBANA_URL")

# Initialize Elasticsearch client
es = Elasticsearch(ELASTICSEARCH_URL, api_key=ELASTIC_API_KEY)

# %% [markdown]
# ## Create Inference Endpoint for Embeddings
#
# We'll use **Jina Embeddings v3** through Elastic Inference Service (EIS). This model:
#
# - Is multilingual out of the box
# - Runs on Elastic's GPU infrastructure (no ML nodes needed)
#
# ### Best Practice: Use EIS for Production Workloads
# EIS eliminates the need to manage ML infrastructure while providing GPU-accelerated performance.

# %%
INFERENCE_ENDPOINT_ID = "embeddings-endpoint"

# Create the inference endpoint for Jina Embeddings v3
inference_config = {
    "service": "elastic",
    "service_settings": {"model_id": "jina-embeddings-v3"},
}

try:
    response = es.inference.put(
        inference_id=INFERENCE_ENDPOINT_ID,
        task_type="text_embedding",
        body=inference_config,
    )

    print(f"Created inference endpoint: {INFERENCE_ENDPOINT_ID}")
    print(json.dumps(response.body, indent=2))
except Exception as e:
    print(f"Error: {e}")

# %%
# Test the inference endpoint
test_response = es.inference.inference(
    inference_id=INFERENCE_ENDPOINT_ID,
    body={
        "input": "Elasticsearch is a distributed search and analytics engine.",
        "input_type": "ingest",
    },
)

embedding = test_response.body["text_embedding"][0]["embedding"]
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# %% [markdown]
# ## Create an Optimized Index for Vector Search
#
# ### Best Practices for Index Design:
# 1. **Use `semantic_text` field type** - Automatically handles chunking and embedding generation
# 2. **Use `copy_to` pattern** - Keep original fields for BM25 and copy content to a dedicated semantic field
# 3. **Consider field lengths** - Jina v3 performs optimally with 2048-4096 tokens
#

# %%
INDEX_NAME = "tech-articles"

# Create index with semantic_text field using copy_to pattern
# Best Practice: Keep original fields for BM25 search and use copy_to for semantic search
index_mappings = {
    "mappings": {
        "properties": {
            "title": {"type": "text", "copy_to": "semantic_field"},
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
if not es.indices.exists(index=INDEX_NAME):
    response = es.indices.create(index=INDEX_NAME, body=index_mappings)
    print(f"Created index: {INDEX_NAME}")
    print(json.dumps(response.body, indent=2))
else:
    print(f"Index '{INDEX_NAME}' already exists, skipping creation")

# %% [markdown]
# ## Ingest Sample Data
#
# ### Best Practice: Use Bulk Operations

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
        es,
        build_data("dataset.json", INDEX_NAME),
    )
    print(f"{success} documents indexed successfully")

    if failed:
        print(f"Errors: {failed}")
except Exception as e:
    print(f"Error: {str(e)}")

# %%
# Check document count
count = es.count(index=INDEX_NAME)
print(f"Total documents in index: {count.body['count']}")

# %% [markdown]
# ## Semantic Search
#
# ### Best Practice: Use `match` Query on `semantic_text` Fields
# The `match` query automatically detects `semantic_text` fields and handles embedding generation for your search query.


# %%
def semantic_search(query: str, size: int = 3):
    """Perform semantic search on the tech-articles index."""
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {"match": {"semantic_field": {"query": query}}},
            "size": size,
            "_source": ["title", "category", "content"],
        },
    )

    print(f"Query: '{query}'\n")
    print(f"Found {response.body['hits']['total']['value']} results:\n")

    for hit in response.body["hits"]["hits"]:
        print(f"Score: {hit['_score']:.4f}")
        print(f"Title: {hit['_source']['title']}")
        print(f"Category: {hit['_source']['category']}")
        print(f"Content: {hit['_source']['content'][:150]}...\n")

    return response


# %%
# Test semantic search
semantic_search("How do I implement similarity search in my application?")

# %% [markdown]
# ## Hybrid Search (BM25 + Semantic)
#
# ### Best Practice: Combine Lexical and Semantic Search
# Hybrid search gives you the best of both worlds - exact keyword matching and semantic understanding.


# %%
def hybrid_search(query: str, size: int = 3):
    """Perform hybrid search combining BM25 and semantic search."""
    response = es.search(
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
                                        "fields": [
                                            "title^2",
                                            "content",
                                        ],  # BM25 on original text fields
                                    }
                                }
                            }
                        },
                        {
                            "standard": {
                                "query": {"match": {"semantic_field": {"query": query}}}
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

    print(f"Hybrid Search Query: '{query}'\n")

    for hit in response.body["hits"]["hits"]:
        print(
            f"Score: {hit['_score']:.4f} | {hit['_source']['title']} [{hit['_source']['category']}]"
        )

    return response


# %%
hybrid_search("What are the best practices for semantic search in Elasticsearch?")

# %% [markdown]
# ## Step 6: Define a Search Workflow
#
# **Elastic Workflows** is an automation engine built into Elasticsearch that orchestrates multi-step processes using YAML.
#
# ### Why Workflows?
# - Automate repeatable search and data processing tasks
# - Chain Elasticsearch operations with flow control (`if`, `foreach`)
# - Serve as reliable, auditable "hands" for AI agents
#
# > **Note:** Workflows is disabled by default. Enable it in Kibana Advanced Settings `Stack Management > Advanced Settings > workflows:ui:enabled`.
# > Workflows are created and managed from the Kibana UI — paste the YAML below into the editor to create this workflow.

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

# %% [markdown]
# ## Step 7: Create an AI Agent with Agent Builder
#
# Agent Builder uses LLMs to power agent reasoning and decision-making.
#
# ### Agents + Workflows
# Agents and Workflows are complementary: workflows handle deterministic, repeatable tasks reliably, while agents provide reasoning for open-ended questions. An agent can invoke a workflow as a tool via MCP, delegating concrete execution while it focuses on understanding the user's intent.
#
# ### Default vs Custom LLM
# - **Elastic Managed LLM** (default): Available out-of-the-box on Elastic Cloud. No configuration or API keys needed.
# - **Custom LLM**: You can configure third-party providers (OpenAI, Azure, Anthropic, etc.) using connectors.
#
# ### SSL Certificate Verification
# For production environments, use `verify=True` when making HTTPS requests:
# - **Elastic Cloud**: Certificates are valid by default, no additional configuration needed.
# - **Self-managed**: You may need to provide the CA certificate path (e.g., `verify="/path/to/ca.crt"`).

# %%
headers = {
    "kbn-xsrf": "true",
    "Authorization": f"ApiKey {ELASTIC_API_KEY}",
    "Content-Type": "application/json",
}

agent_payload = {
    "id": "tech-articles-assistant",
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
    agent_id = agent_data.get("id")
    print(f"Created agent: {agent_id}")
    print(json.dumps(agent_data, indent=2))
else:
    print(f"Error creating agent: {response.text}")
    agent_id = None


# %%
# Chat with the agent
def chat_with_agent(agent_id: str, message: str):
    """Send a message to the agent and get a response."""
    chat_payload = {"input": message, "agent_id": agent_id}

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
if agent_id:
    result = chat_with_agent(
        agent_id, "What are the best practices for building RAG applications?"
    )
    print(json.dumps(result, indent=2))

# %%
# es.indices.delete(index=INDEX_NAME)
# es.inference.delete(inference_id=INFERENCE_ENDPOINT_ID)

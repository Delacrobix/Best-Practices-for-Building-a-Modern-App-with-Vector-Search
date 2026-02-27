```mermaid
graph TD
    HN["Hacker News API"] -->|stories| WF["Elastic Workflow"]
    WF -->|index documents| ES["Elasticsearch Index"]
    ES -->|auto-embed via| EIS["EIS (Jina v3)"]
    WF -->|invoke| Agent["AI Agent"]
    Agent -->|search| ES
```
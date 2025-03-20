# PDF Knowledge Graph Extractor and Visualizer

This Streamlit application extracts and visualizes knowledge graphs from PDF documents using an agent-based approach with AI language models.

## Features

- PDF text extraction and processing
- Multi-agent architecture with specialized roles:
  - Entity extraction agent
  - Relationship extraction agent
  - JSON-LD generation agent
- Knowledge graph visualization with ECharts
- Support for multiple AI APIs (OpenAI and Groq)
- Customizable extraction prompts
- Intermediate result viewing and export

## Requirements

The project requires the following Python packages:
```
streamlit==1.27.0
PyPDF2==3.0.1
chardet==5.2.0
streamlit-echarts==0.4.0
langchain-text-splitters==0.0.1
nltk==3.8.1
openai==1.3.0
groq==0.4.0
```

## How to Use

1. Upload a PDF document in the first tab
2. Configure extraction settings in the second tab
   - Choose API provider (OpenAI or Groq)
   - Enter your API key
   - Customize extraction prompts if needed
3. Start the extraction process
4. View the knowledge graph visualization
5. Examine intermediate extraction results (entities and relationships)
6. Access the raw JSON-LD data

## Architecture

The application follows an agent-based architecture:
- **PDF Chunking Agent**: Splits PDF content into manageable chunks
- **Entity Extraction Agent**: Identifies entities in the text chunks
- **Relationship Extraction Agent**: Finds relationships between entities
- **JSON Generation Agent**: Creates structured JSON-LD representation
- **Workflow Manager**: Coordinates the agent workflow


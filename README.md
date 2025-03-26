# AI-AGENTIC Knowledge Graph Extractor

## Overview

AI-AGENTIC Knowledge Graph Extractor is a Streamlit-based application designed for extracting and visualizing knowledge graphs from PDF documents. The project leverages large language models (LLMs) to extract entities and relationships, generating structured knowledge representations in JSON-LD format.

## Features

- Upload PDFs: Extract text using PyPDF2.
- Customizable Prompts: Upload text prompts for entity and relationship extraction.
- Chunking: Break down large text into manageable parts for better LLM processing.
- Entity & Relationship Extraction: Automatically identify and categorize materials, processes, properties, and relationships.
- JSON-LD Generation: Output structured JSON-LD files that are ready for visualization.
- API Integration: Supports both OpenAI and Groq APIs for LLM functionality.
- Chain of Thought: Provide better response from LLM


## File Structure

```
.
├── Prompts/                       # Folder containing prompt templates
│   ├── Prompt Relationship extraction.txt
│   ├── Prompt entity extraction.txt
│   ├── prompt json.txt
├── Sample/                        # Example input and output files
│   ├── Sample.pdf                 # Example PDF file for processing
├── README.md                      # Documentation
├── Requirement                    # Requirements file
├── mini.py                        # Main application script
├── kgvis.py                       # Knowledge graph visualization module
```

## Requirements

- Python 3.8+
- Libraries:
  - streamlit
  - PyPDF2
  - langchain-text-splitters
  - openai
  - chardet
  - groq

To install dependencies, use:
```bash
pip install -r requirements.txt
```

## Installation Process

1. Clone the repository:
   ```bash
   git clone https://github.com/shubhamnehate/AI-AGENTIC-KG-EXTRACTOR.git
   cd AI-AGENTIC-KG-EXTRACTOR
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run mini.py
   ```

1. Upload a PDF file using the interface.
2. Select your preferred API (OpenAI or Groq) and provide an API key.
3. Upload prompt files for:
   - Entity Extraction
   - Relationship Extraction
   - JSON-LD Generation
4. Process the PDF to generate knowledge graphs in JSON-LD format

### Example Output

```json
{
  "@context": "http://emmo.info/emmo#",
  "@graph": [
    {
      "@id": "ex:Material_1",
      "@type": "Material",
      "has_property": {
        "@id": "ex:Property_1",
        "value": "520 nm"
      }
    }
  ]
}
```

## Thanks 🙏
This work would not have been possible without amazing open source models and datasets, including (but not limited to):


   - [KGVIS By Dr. Mohammad J. Eslamibidgoli](https://scholar.google.ca/citations?user=4flM9wEAAAAJ&hl=en)
   - [Data-extraction by Nasrin mohammadi](https://github.com/Nasrin-mohammadi/data-extraction) 📄
   - This project uses the EMMO Ontology for knowledge graph structuring and builds on existing research in materials science and hydrogen technologies.
   - [Groq](https://groq.com) 🤖
   - [OpenAI](https://platform.openai.com/docs/api-reference/introduction) 🤖
     

Thank you to the authors of these tools, models, and datasets for making them available to the community!


## Acknowledgements 💌
Special thanks to **FZJ IET3** and the **AMI Division** for their invaluable support and guidance in this project.



## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.


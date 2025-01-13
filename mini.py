# #  pip install --upgrade streamlit PyPDF2 langchain langchain-text-splitters tiktoken nltk openai groq
# # Run using streamlit run mini.py

import streamlit as st
from PyPDF2 import PdfReader
import json
from langchain_text_splitters import NLTKTextSplitter
import openai
from groq import Groq

def truncate_conversation(conversation, max_tokens=120000):
    """Truncate conversation to fit within context length."""
    total_tokens = sum(len(item["content"].split()) for item in conversation if item["content"])
    while total_tokens > max_tokens and len(conversation) > 1:
        conversation.pop(0)  # Remove the oldest message
        total_tokens = sum(len(item["content"].split()) for item in conversation if item["content"])

def call_openai_api(content, api_key, model="gpt-4o"):
    """Simplified OpenAI API call without persistent conversation."""
    client = openai.OpenAI(api_key=api_key)
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def call_groq_api(prompt, api_key, model="llama3-8b-8192"):
    """Make an API call to Groq."""
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model
        )
    return chat_completion.choices[0].message.content

def unified_api_call(api_choice, prompt, api_key, model):
    """Unified API call function."""
    if api_choice == "OpenAI API":
        return call_openai_api(prompt, api_key, model=model)
    elif api_choice == "Groq API":
        return call_groq_api(prompt, api_key, model=model)
    else:
        st.error("Invalid API Choice")
        return None

# Sidebar for API Selection and Prompt Upload
st.sidebar.title("API Selection and Prompts")
api_choice = st.sidebar.radio("Choose API", ("OpenAI API", "Groq API"))
api_key = st.sidebar.text_input("Enter your API Key", type="password")
model = st.sidebar.text_input("Enter Model Name", value="gpt-4o")

# Prompt Uploads
entity_prompt_file = st.sidebar.file_uploader("Upload Entity Extraction Prompt", type="txt", key="entity_prompt")
relationship_prompt_file = st.sidebar.file_uploader("Upload Relationship Extraction Prompt", type="txt", key="relationship_prompt")
json_prompt_file = st.sidebar.file_uploader("Upload JSON-LD Generation Prompt", type="txt", key="json_prompt")

# File uploader for PDF input
st.title("Enhanced Streamlit LLM Knowledge Graph Generator")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def read_prompt_file(prompt_file, default_prompt):
    if prompt_file is not None:
        try:
            return prompt_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading prompt file: {e}")
    return default_prompt

def chunk_text(pdf_content):
    """Chunk text using NltkTextSplitter."""
    splitter = NLTKTextSplitter(chunk_size=3000, chunk_overlap=100)
    chunks = splitter.split_text(pdf_content)
    return chunks

def process_in_chunks(text, max_length=4000):
    """Split text into smaller chunks for processing."""
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind("\n")  # Split at the last newline
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks

class AIAgent:
    def __init__(self, name, task_function):
        self.name = name
        self.task_function = task_function

    def execute(self, *args, **kwargs):
        st.write(f"Executing {self.name}...")
        result = self.task_function(*args, **kwargs)
        if isinstance(result, str):
            st.download_button(
                label=f"Download {self.name} Output",
                data=result,
                file_name=f"{self.name.replace(' ', '_').lower()}_output.txt",
                mime="text/plain"
            )
        elif isinstance(result, dict) or isinstance(result, list):
            st.download_button(
                label=f"Download {self.name} Output",
                data=json.dumps(result, indent=4),
                file_name=f"{self.name.replace(' ', '_').lower()}_output.json",
                mime="application/json"
            )
        return result

def agent_chunking_task(pdf_content):
    return chunk_text(pdf_content)

def agent_entity_extraction_task(chunks):
    default_prompt = "Extract entities from the following text:"  # Default prompt for entity extraction
    prompt = read_prompt_file(entity_prompt_file, default_prompt)
    all_entities = []
    for i, chunk in enumerate(chunks):
        response = unified_api_call(api_choice, prompt + "\n\n" + chunk, api_key, model)
        if response:
            all_entities.append(response.strip())
    combined_prompt = "Condense the following extracted entities into a single output without missing any unique information:\n" + "\n".join(all_entities)
    final_entities = []
    for sub_prompt in process_in_chunks(combined_prompt):
        response = unified_api_call(api_choice, sub_prompt, api_key, model)
        if response:
            final_entities.append(response.strip())
    final_combined_prompt = "Condense the following condensed parts into one final output:\n" + "\n".join(final_entities)
    final_response = unified_api_call(api_choice, final_combined_prompt, api_key, model)
    return final_response

def agent_relationship_extraction_task(chunks, entity_extractions):
    default_prompt = "Extract relationships between entities for the following text and entities:"  # Default prompt for relationship extraction
    prompt = read_prompt_file(relationship_prompt_file, default_prompt)
    relationships = []
    for i, (chunk, entity_extraction) in enumerate(zip(chunks, entity_extractions)):
        input_text = prompt + f"\n\nText: {chunk}\nEntities: {entity_extraction}"
        response = unified_api_call(api_choice, input_text, api_key, model)
        if response:
            relationships.append(response.strip())
    combined_prompt = "Condense the following extracted relationships into a single output without missing any unique information:\n" + "\n".join(relationships)
    final_relationships = []
    for sub_prompt in process_in_chunks(combined_prompt):
        response = unified_api_call(api_choice, sub_prompt, api_key, model)
        if response:
            final_relationships.append(response.strip())
    final_combined_prompt = "Condense the following condensed parts into one final output:\n" + "\n".join(final_relationships)
    final_response = unified_api_call(api_choice, final_combined_prompt, api_key, model)
    return final_response

def agent_json_generation_task(entity_extractions, relationships):
    default_prompt = "Generate a JSON-LD representation for the following entities and relationships:"  # Default prompt for JSON generation
    prompt = read_prompt_file(json_prompt_file, default_prompt)

    # Pass condensed data directly
    json_ld_prompt = (
        prompt + "DATA (START)\n\nEntities: " + entity_extractions + "\nRelationships: " + relationships + "\n DATA (END)"
    )
    response = unified_api_call(api_choice, json_ld_prompt, api_key, model)
    return response

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        pdf_content = "".join(page.extract_text() for page in reader.pages)
        st.write("PDF content extracted successfully.")
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        pdf_content = None

    if pdf_content:
        chunking_agent = AIAgent("Chunking Agent", agent_chunking_task)
        entity_extraction_agent = AIAgent("Entity Extraction Agent", agent_entity_extraction_task)
        relationship_extraction_agent = AIAgent("Relationship Extraction Agent", agent_relationship_extraction_task)
        json_generation_agent = AIAgent("JSON Generation Agent", agent_json_generation_task)

        # Run chunking agent
        chunks = chunking_agent.execute(pdf_content)
        st.write(f"Total chunks created: {len(chunks)}")

        # Run entity extraction agent
        entity_extractions = entity_extraction_agent.execute(chunks)
        st.write("Final Extracted Entities:")
        st.write(entity_extractions)

        # Run relationship extraction agent
        relationships = relationship_extraction_agent.execute(chunks, entity_extractions)
        st.write("Final Extracted Relationships:")
        st.write(relationships)

        # Run JSON generation agent
        json_ld_output = json_generation_agent.execute(entity_extractions, relationships)
        st.write("Generated JSON-LD:")
        st.write(json_ld_output)

        
else:
    st.warning("Please upload a PDF file to proceed.")

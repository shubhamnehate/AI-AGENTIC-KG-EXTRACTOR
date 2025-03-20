import streamlit as st
import os
import json
import time
from streamlit_echarts import st_echarts

# Import utility modules
from utils.pdf_processor import extract_text_from_pdf, chunk_text, process_in_chunks, read_prompt_file
from utils.api_clients import truncate_conversation, unified_api_call
from utils.graph_utils import extract_graph_data, create_echarts_option
from utils.json_validator import read_json_ld, validate_knowledge_graph_json, extract_json_from_text

# Import agent-based workflow
from agents import WorkflowManager

# Configure page
st.set_page_config(layout="wide", page_title="PDF to Knowledge Graph")
st.title("PDF Knowledge Graph Extractor and Visualizer")
st.markdown("Extract entities and relationships from PDFs and visualize as a knowledge graph")

# Initialize session state variables if they don't exist
if 'pdf_content' not in st.session_state:
    st.session_state.pdf_content = None
if 'knowledge_graph_data' not in st.session_state:
    st.session_state.knowledge_graph_data = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'current_chunk_index' not in st.session_state:
    st.session_state.current_chunk_index = 0
if 'extraction_status' not in st.session_state:
    st.session_state.extraction_status = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_choice' not in st.session_state:
    st.session_state.api_choice = "OpenAI API"
if 'model' not in st.session_state:
    st.session_state.model = "gpt-4o"

# Entity extraction prompt
if 'entity_prompt' not in st.session_state:
    st.session_state.entity_prompt = """
You are a Materials Science expert. Extract entities from the text using these categories:
- Material: any material, substance, or chemical
- Manufacturing: any manufacturing process, synthesis, or fabrication
- Measurement: any characterization, measurement, or analysis
- Property: any property, characteristic, or attribute
- Parameter: any processing parameter, condition, or variable

Format your response as a structured list of entities.
"""

# Relationship extraction prompt
if 'relationship_prompt' not in st.session_state:
    st.session_state.relationship_prompt = """
You are a Materials Science expert. Extract relationships between the entities using these relationship types:
- is_manufacturing_input: material → manufacturing
- has_manufacturing_output: manufacturing → material
- is_measurement_input: material → measurement
- has_measurement_output: measurement → property
- has_property: material → property
- has_parameter: manufacturing/measurement → parameter

Format your response as a simple list of relationships.
"""

# JSON-LD generation prompt
if 'json_prompt' not in st.session_state:
    st.session_state.json_prompt = """
Generate a JSON-LD representation of the following entities and relationships.

Format:
{
  "@context": {
    "ex": "http://example.com/",
    "emmo": "http://emmo.info/emmo#",
    "skos": "http://www.w3.org/2004/02/skos/core#"
  },
  "@graph": [
    {
      "@id": "ex:Material1",
      "@type": "emmo:EMMO_4207e895_8b83_4318_996a_72cfb32acd94",
      "skos:prefLabel": "Material Name"
    },
    ...
  ]
}

Use these entity types:
- Material: emmo:EMMO_4207e895_8b83_4318_996a_72cfb32acd94
- Manufacturing: emmo:EMMO_a4d66059_5dd3_4b90_b4cb_10960559441b
- Measurement: emmo:EMMO_463bcfda_867b_41d9_a967_211d4d437cfb
- Property: emmo:EMMO_b7bcff25_ffc3_474e_9ab5_01b1664bd4ba
- Parameter: emmo:EMMO_d1d436e7_72fc_49cd_863b_7bfb4ba5276a

And these relationship types:
- is_manufacturing_input: emmo:EMMO_e1097637
- has_manufacturing_output: emmo:EMMO_e1245987
- is_measurement_input: emmo:EMMO_m5677989
- has_measurement_output: emmo:EMMO_m87987545
- has_property: emmo:EMMO_p5778r78
- has_parameter: emmo:EMMO_p46903ar7
"""

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'nodes' not in st.session_state:
    st.session_state.nodes = []
if 'links' not in st.session_state:
    st.session_state.links = []
if 'layout' not in st.session_state:
    st.session_state.layout = 'force'
if 'hide_units_and_literals' not in st.session_state:
    st.session_state.hide_units_and_literals = False
if 'previous_tab' not in st.session_state:
    st.session_state.previous_tab = None
if 'entities' not in st.session_state:
    st.session_state.entities = ""
if 'relationships' not in st.session_state:
    st.session_state.relationships = ""

# Define our tabs for different stages of the process
tabs = st.tabs(["PDF Upload", "Extraction Settings", "Knowledge Graph", "Intermediate Results", "Raw JSON-LD Data"])

# PDF Upload Tab
with tabs[0]:
    st.header("Upload PDF Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Process the uploaded PDF
        try:
            with st.spinner("Extracting text from PDF..."):
                pdf_content = extract_text_from_pdf(uploaded_file)
                st.session_state.pdf_content = pdf_content
                st.success(f"Successfully extracted {len(pdf_content)} characters from PDF")
                
                # Display a preview of the extracted text
                st.subheader("PDF Content Preview")
                st.text_area("Extracted text", value=pdf_content[:1500] + "..." if len(pdf_content) > 1500 else pdf_content, height=300)
                
                # Chunk the text for easier processing
                chunks = chunk_text(pdf_content, st.session_state.api_choice)
                st.session_state.chunks = chunks
                st.session_state.current_chunk_index = 0
                
                st.info(f"PDF content split into {len(chunks)} chunks for processing")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Extraction Settings Tab
with tabs[1]:
    st.header("Knowledge Graph Extraction Settings")
    
    # API selection
    api_choice = st.radio(
        "Choose AI API Provider:",
        ["OpenAI API", "Groq API"],
        index=0 if st.session_state.api_choice == "OpenAI API" else 1
    )
    st.session_state.api_choice = api_choice
    
    # Model selection based on API provider
    if api_choice == "OpenAI API":
        model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        model = st.selectbox("Choose OpenAI model:", model_options, index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0)
    else:  # Groq API
        model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
        model = st.selectbox("Choose Groq model:", model_options, index=model_options.index(st.session_state.model) if st.session_state.model in model_options else 0)
    st.session_state.model = model
    
    # API key input
    api_key = st.text_input("Enter API key:", value=st.session_state.api_key, type="password")
    st.session_state.api_key = api_key
    
    # Prompt customization
    st.subheader("Extraction Prompts")
    
    # Create tabs for different prompt types
    prompt_tabs = st.tabs(["Entity Extraction", "Relationship Extraction", "JSON-LD Generation"])
    
    # Entity Extraction Prompt
    with prompt_tabs[0]:
        entity_prompt_file = st.file_uploader("Upload a custom entity extraction prompt (optional)", type=["txt"], key="entity_prompt_file")
        
        if entity_prompt_file:
            custom_entity_prompt = read_prompt_file(entity_prompt_file, st.session_state.entity_prompt)
            st.session_state.entity_prompt = custom_entity_prompt
        
        st.text_area("Edit entity extraction prompt:", value=st.session_state.entity_prompt, height=200, key="entity_prompt_area")
        if st.session_state.entity_prompt != st.session_state["entity_prompt_area"]:
            st.session_state.entity_prompt = st.session_state["entity_prompt_area"]
    
    # Relationship Extraction Prompt
    with prompt_tabs[1]:
        relationship_prompt_file = st.file_uploader("Upload a custom relationship extraction prompt (optional)", type=["txt"], key="relationship_prompt_file")
        
        if relationship_prompt_file:
            custom_relationship_prompt = read_prompt_file(relationship_prompt_file, st.session_state.relationship_prompt)
            st.session_state.relationship_prompt = custom_relationship_prompt
        
        st.text_area("Edit relationship extraction prompt:", value=st.session_state.relationship_prompt, height=200, key="relationship_prompt_area")
        if st.session_state.relationship_prompt != st.session_state["relationship_prompt_area"]:
            st.session_state.relationship_prompt = st.session_state["relationship_prompt_area"]
    
    # JSON-LD Generation Prompt
    with prompt_tabs[2]:
        json_prompt_file = st.file_uploader("Upload a custom JSON-LD generation prompt (optional)", type=["txt"], key="json_prompt_file")
        
        if json_prompt_file:
            custom_json_prompt = read_prompt_file(json_prompt_file, st.session_state.json_prompt)
            st.session_state.json_prompt = custom_json_prompt
        
        st.text_area("Edit JSON-LD generation prompt:", value=st.session_state.json_prompt, height=200, key="json_prompt_area")
        if st.session_state.json_prompt != st.session_state["json_prompt_area"]:
            st.session_state.json_prompt = st.session_state["json_prompt_area"]
    
    # Start extraction process using agent-based workflow
    if st.button("Start Knowledge Graph Extraction"):
        if not st.session_state.pdf_content:
            st.error("Please upload a PDF document first")
        elif not st.session_state.api_key:
            st.error("Please enter a valid API key")
        else:
            # Reset previous results
            st.session_state.knowledge_graph_data = None
            st.session_state.processing_complete = False
            st.session_state.extraction_status = "Processing"
            st.session_state.current_chunk_index = 0
            
            # Process chunks sequentially with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create the workflow manager
                workflow_manager = WorkflowManager()
                
                # Define progress callback
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                # Define status callback
                def update_status(status_message):
                    status_text.text(status_message)
                    
                # Process the PDF using the agent-based workflow
                result = workflow_manager.process_pdf(
                    pdf_content=st.session_state.pdf_content,
                    entity_prompt=st.session_state.entity_prompt,
                    relationship_prompt=st.session_state.relationship_prompt,
                    json_prompt=st.session_state.json_prompt,
                    api_choice=st.session_state.api_choice,
                    api_key=st.session_state.api_key,
                    model=st.session_state.model,
                    hide_units_and_literals=st.session_state.hide_units_and_literals,
                    progress_callback=update_progress,
                    status_callback=update_status
                )
                
                if result["success"]:
                    # Store the results
                    st.session_state.knowledge_graph_data = result["knowledge_graph"]
                    st.session_state.nodes = result["nodes"]
                    st.session_state.links = result["links"]
                    st.session_state.entities = result["entities"]
                    st.session_state.relationships = result["relationships"]
                    
                    st.session_state.processing_complete = True
                    st.session_state.extraction_status = "Complete"
                    
                    # Update the progress bar to 100%
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Success message with stats
                    st.success(f"Successfully extracted knowledge graph with {len(st.session_state.nodes)} nodes and {len(st.session_state.links)} relationships")
                else:
                    st.error(f"Failed to extract knowledge graph: {result['message']} (Stage: {result['stage']})")
                    st.session_state.extraction_status = "Failed"
            except Exception as e:
                st.error(f"Error during knowledge graph extraction: {str(e)}")
                st.session_state.extraction_status = "Failed"
    
    # Show current status if a process is running
    if st.session_state.extraction_status:
        st.info(f"Extraction Status: {st.session_state.extraction_status}")
        
        if st.session_state.extraction_status == "Processing":
            st.info(f"Processing chunk {st.session_state.current_chunk_index + 1} of {len(st.session_state.chunks)}")

# Knowledge Graph Visualization Tab
with tabs[2]:
    st.header("Knowledge Graph Visualization")
    
    if st.session_state.processing_complete and st.session_state.nodes and st.session_state.links:
        # Visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            layout = st.selectbox(
                "Choose graph layout:",
                ["force", "circular", "none"],
                index=["force", "circular", "none"].index(st.session_state.layout)
            )
            st.session_state.layout = layout
        
        with col2:
            hide_units = st.checkbox(
                "Hide units and literal values (simpler view)", 
                value=st.session_state.hide_units_and_literals
            )
            if hide_units != st.session_state.hide_units_and_literals:
                st.session_state.hide_units_and_literals = hide_units
                # Re-extract the graph data with the new setting
                st.session_state.nodes, st.session_state.links = extract_graph_data(
                    st.session_state.knowledge_graph_data, 
                    st.session_state.hide_units_and_literals
                )
        
        # Display graph statistics
        st.info(f"Graph contains {len(st.session_state.nodes)} nodes and {len(st.session_state.links)} relationships")
        
        # Create and display the ECharts graph
        options = create_echarts_option(st.session_state.nodes, st.session_state.links, st.session_state.layout)
        st_echarts(options=options, height="800px")
        
        # Download options
        st.download_button(
            "Download Graph Data as JSON",
            data=json.dumps({
                "nodes": st.session_state.nodes,
                "links": st.session_state.links
            }, indent=2),
            file_name="knowledge_graph_data.json",
            mime="application/json"
        )
    else:
        st.info("No knowledge graph data available. Please upload a PDF and extract knowledge graphs in the previous tabs.")

# Intermediate Results Tab
with tabs[3]:
    st.header("Intermediate Extraction Results")
    
    if st.session_state.processing_complete and (st.session_state.entities or st.session_state.relationships):
        # Display intermediate extraction results
        st.subheader("Extracted Entities")
        st.text_area("Entities", value=st.session_state.entities, height=300, disabled=True)
        
        st.subheader("Extracted Relationships")
        st.text_area("Relationships", value=st.session_state.relationships, height=300, disabled=True)
        
        # Add download options for the intermediate results
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "Download Entities",
                data=st.session_state.entities,
                file_name="extracted_entities.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                "Download Relationships",
                data=st.session_state.relationships,
                file_name="extracted_relationships.txt",
                mime="text/plain"
            )
    else:
        st.info("No intermediate extraction results available. Please extract knowledge graphs in the previous tabs.")

# Raw JSON-LD Data Tab
with tabs[4]:
    st.header("Raw JSON-LD Data")
    
    if st.session_state.knowledge_graph_data:
        # Display the raw JSON-LD data
        st.json(st.session_state.knowledge_graph_data)
        
        # Download option
        st.download_button(
            "Download Raw JSON-LD",
            data=json.dumps(st.session_state.knowledge_graph_data, indent=2),
            file_name="knowledge_graph.jsonld",
            mime="application/ld+json"
        )
    else:
        st.info("No JSON-LD data available. Please extract knowledge graphs in the previous tabs.")

# Track the current tab to avoid unnecessary recomputation
current_tab_index = st.session_state.get("current_tab_index", 0)
st.session_state.current_tab_index = current_tab_index

# Footer with instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Upload a PDF document in the first tab
2. Configure extraction settings in the second tab
3. View the knowledge graph visualization in the third tab
4. Examine intermediate extraction results (entities and relationships) in the fourth tab
5. Access the raw JSON-LD data in the fifth tab

**Note:** For large PDFs, the extraction process may take some time. The document is processed in chunks.
""")

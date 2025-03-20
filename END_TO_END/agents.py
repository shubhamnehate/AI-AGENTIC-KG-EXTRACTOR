import streamlit as st
import time
from utils.pdf_processor import chunk_text, process_in_chunks, read_prompt_file
from utils.api_clients import unified_api_call
from utils.json_validator import extract_json_from_text, validate_knowledge_graph_json
from utils.graph_utils import extract_graph_data

class Agent:
    """Base agent class for handling specific tasks in the workflow."""
    def __init__(self, name):
        self.name = name
    
    def execute(self):
        """Execute the agent's task. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

class PDFChunkingAgent(Agent):
    """Agent responsible for chunking PDF content."""
    def __init__(self):
        super().__init__("PDF Chunking Agent")
    
    def execute(self, pdf_content, api_choice="OpenAI API"):
        """Chunk the PDF content into manageable pieces."""
        try:
            chunks = chunk_text(pdf_content, api_choice)
            return {
                "success": True,
                "chunks": chunks,
                "message": f"Split PDF into {len(chunks)} chunks"
            }
        except Exception as e:
            return {
                "success": False,
                "chunks": [],
                "message": f"Error chunking PDF: {str(e)}"
            }

class EntityExtractionAgent(Agent):
    """Agent responsible for extracting entities from text chunks."""
    def __init__(self):
        super().__init__("Entity Extraction Agent")
    
    def execute(self, chunks, entity_prompt, api_choice, api_key, model, **kwargs):
        """Extract entities from PDF text chunks."""
        progress_callback = kwargs.get('progress_callback', None)
        status_callback = kwargs.get('status_callback', None)
        
        try:
            if status_callback:
                status_callback("Extracting entities...")
            
            # For Groq API, use a simplified prompt to save tokens
            if api_choice == "Groq API":
                # Extract just the essential instructions from the prompt
                simplified_prompt = """
                You are a Materials Science expert. Extract entities from the text using these categories:
                - Material: any material, substance, or chemical
                - Manufacturing: any manufacturing process, synthesis, or fabrication
                - Measurement: any characterization, measurement, or analysis
                - Property: any property, characteristic, or attribute
                - Parameter: any processing parameter, condition, or variable
                
                Format your response as a structured list of entities.
                """
                entity_prompt = simplified_prompt
            
            total_chunks = len(chunks)
            all_entities = []
            
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    # Ensure progress is between 0.0 and 0.2 for this phase
                    progress_value = i / total_chunks * 0.2
                    progress_callback(min(0.2, progress_value))
                if status_callback:
                    status_callback(f"Extracting entities from chunk {i+1} of {total_chunks}...")
                    
                input_text = entity_prompt + f"\n\nText: {chunk}"
                response = unified_api_call(api_choice, input_text, api_key, model)
                
                if response:
                    all_entities.append(response.strip())
                    
            # Combine entity results
            if status_callback:
                status_callback("Condensing entity results...")
                
            combined_prompt = "Condense the following extracted entities into a single output without missing any unique information:\n" + "\n".join(all_entities)
            final_entities = []
            
            for i, sub_prompt in enumerate(process_in_chunks(combined_prompt, api_choice=api_choice)):
                if progress_callback:
                    # Ensure progress is between 0.2 and 0.4 for this condensing phase
                    chunk_count = len(list(process_in_chunks(combined_prompt, api_choice=api_choice)))
                    progress_value = 0.2 + (i / max(1, chunk_count) * 0.2)
                    progress_callback(min(0.4, progress_value))
                response = unified_api_call(api_choice, sub_prompt, api_key, model)
                if response:
                    final_entities.append(response.strip())
                    
            final_combined_prompt = "Condense the following condensed parts into one final output:\n" + "\n".join(final_entities)
            final_response = unified_api_call(api_choice, final_combined_prompt, api_key, model)
            
            return {
                "success": bool(final_response),
                "entities": final_response,
                "message": "Entity extraction complete"
            }
        except Exception as e:
            return {
                "success": False,
                "entities": "",
                "message": f"Error in entity extraction: {str(e)}"
            }

class RelationshipExtractionAgent(Agent):
    """Agent responsible for extracting relationships between entities."""
    def __init__(self):
        super().__init__("Relationship Extraction Agent")
    
    def execute(self, chunks, entities, relationship_prompt, api_choice, api_key, model, **kwargs):
        """Extract relationships between extracted entities."""
        progress_callback = kwargs.get('progress_callback', None)
        status_callback = kwargs.get('status_callback', None)
        
        try:
            if status_callback:
                status_callback("Extracting relationships...")
            
            # For Groq API, use a simplified prompt to save tokens
            if api_choice == "Groq API":
                # Extract just the essential instructions from the prompt
                simplified_prompt = """
                You are a Materials Science expert. Extract relationships between the entities using these relationship types:
                - is_manufacturing_input: material → manufacturing
                - has_manufacturing_output: manufacturing → material
                - is_measurement_input: material → measurement
                - has_measurement_output: measurement → property
                - has_property: material → property
                - has_parameter: manufacturing/measurement → parameter
                
                Format your response as a simple list of relationships.
                """
                relationship_prompt = simplified_prompt
            
            total_chunks = len(chunks)
            relationships = []
            
            for i, chunk in enumerate(chunks):
                if progress_callback:
                    # Ensure progress is between 0.4 and 0.6 for this phase
                    progress_value = 0.4 + (i / total_chunks * 0.2)
                    progress_callback(min(0.6, progress_value))
                if status_callback:
                    status_callback(f"Extracting relationships from chunk {i+1} of {total_chunks}...")
                    
                input_text = relationship_prompt + f"\n\nText: {chunk}\nEntities: {entities}"
                response = unified_api_call(api_choice, input_text, api_key, model)
                
                if response:
                    relationships.append(response.strip())
                    
            # Combine relationship results
            if status_callback:
                status_callback("Condensing relationship results...")
                
            combined_prompt = "Condense the following extracted relationships into a single output without missing any unique information:\n" + "\n".join(relationships)
            final_relationships = []
            
            for i, sub_prompt in enumerate(process_in_chunks(combined_prompt, api_choice=api_choice)):
                if progress_callback:
                    # Ensure progress is between 0.6 and 0.8 for this condensing phase
                    chunk_count = len(list(process_in_chunks(combined_prompt, api_choice=api_choice)))
                    progress_value = 0.6 + (i / max(1, chunk_count) * 0.2)
                    progress_callback(min(0.8, progress_value))
                response = unified_api_call(api_choice, sub_prompt, api_key, model)
                if response:
                    final_relationships.append(response.strip())
                    
            final_combined_prompt = "Condense the following condensed parts into one final output:\n" + "\n".join(final_relationships)
            final_response = unified_api_call(api_choice, final_combined_prompt, api_key, model)
            
            return {
                "success": bool(final_response),
                "relationships": final_response,
                "message": "Relationship extraction complete"
            }
        except Exception as e:
            return {
                "success": False,
                "relationships": "",
                "message": f"Error in relationship extraction: {str(e)}"
            }

class JSONGenerationAgent(Agent):
    """Agent responsible for generating JSON-LD output."""
    def __init__(self):
        super().__init__("JSON Generation Agent")
    
    def execute(self, entities, relationships, json_prompt, api_choice, api_key, model, **kwargs):
        """Generate JSON-LD from entities and relationships."""
        progress_callback = kwargs.get('progress_callback', None)
        status_callback = kwargs.get('status_callback', None)
        
        try:
            if status_callback:
                status_callback("Generating JSON-LD...")
            
            if progress_callback:
                progress_callback(0.85)  # Final progress stage
            
            # For Groq API, use a simplified prompt to save tokens
            if api_choice == "Groq API":
                # Extract just the essential instructions from the prompt
                simplified_prompt = """
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
                json_prompt = simplified_prompt
            
            # Pass condensed data directly
            json_ld_prompt = (
                json_prompt + "\n\nDATA (START)\n\nEntities: " + entities + "\nRelationships: " + relationships + "\nDATA (END)"
            )
            
            # For Groq API, make explicit request for JSON in the prompt
            if api_choice == "Groq API":
                json_ld_prompt += "\n\nReturn ONLY the JSON-LD object with no explanations or markdown formatting."
            else:
                json_ld_prompt += "\n\nRespond with only the JSON-LD object. No other text."
            
            response = unified_api_call(api_choice, json_ld_prompt, api_key, model)
            
            if status_callback:
                status_callback("Validating JSON-LD...")
            
            # Try to parse the JSON-LD from the response
            try:
                # Extract and validate JSON from the response
                json_data = extract_json_from_text(response)
                
                if json_data and validate_knowledge_graph_json(json_data):
                    nodes, links = extract_graph_data(json_data)
                    
                    if progress_callback:
                        progress_callback(1.0)
                    if status_callback:
                        status_callback("JSON-LD generation complete!")
                    
                    return {
                        "success": True,
                        "json_ld": json_data,
                        "nodes": nodes,
                        "links": links,
                        "message": "JSON-LD generation complete"
                    }
                else:
                    return {
                        "success": False,
                        "json_ld": None,
                        "message": "Invalid JSON-LD was generated"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "json_ld": None,
                    "message": f"Error parsing JSON-LD: {str(e)}"
                }
        except Exception as e:
            return {
                "success": False,
                "json_ld": None,
                "message": f"Error generating JSON-LD: {str(e)}"
            }

class WorkflowManager:
    """Manager to coordinate the agents in the extraction workflow."""
    def __init__(self):
        self.chunking_agent = PDFChunkingAgent()
        self.entity_agent = EntityExtractionAgent()
        self.relationship_agent = RelationshipExtractionAgent()
        self.json_agent = JSONGenerationAgent()
    
    def process_pdf(self, 
                   pdf_content, 
                   entity_prompt,
                   relationship_prompt,
                   json_prompt,
                   api_choice, 
                   api_key, 
                   model, 
                   hide_units_and_literals=False, 
                   progress_callback=None, 
                   status_callback=None):
        """Run the complete workflow to process a PDF into a knowledge graph."""
        # Step 1: Chunk the PDF
        chunking_result = self.chunking_agent.execute(pdf_content, api_choice)
        if not chunking_result["success"]:
            return {
                "success": False,
                "message": chunking_result["message"],
                "stage": "chunking"
            }
        
        chunks = chunking_result["chunks"]
        
        # Step 2: Extract entities
        entity_result = self.entity_agent.execute(
            chunks, entity_prompt, api_choice, api_key, model,
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        if not entity_result["success"]:
            return {
                "success": False,
                "message": entity_result["message"],
                "stage": "entity_extraction"
            }
        
        entities = entity_result["entities"]
        
        # Step 3: Extract relationships
        relationship_result = self.relationship_agent.execute(
            chunks, entities, relationship_prompt, api_choice, api_key, model,
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        if not relationship_result["success"]:
            return {
                "success": False,
                "message": relationship_result["message"],
                "stage": "relationship_extraction"
            }
        
        relationships = relationship_result["relationships"]
        
        # Step 4: Generate JSON-LD
        json_result = self.json_agent.execute(
            entities, relationships, json_prompt, api_choice, api_key, model,
            progress_callback=progress_callback,
            status_callback=status_callback
        )
        if not json_result["success"]:
            return {
                "success": False,
                "message": json_result["message"],
                "stage": "json_generation"
            }
        
        # Extract nodes and links with hide_units_and_literals option
        if hide_units_and_literals:
            nodes, links = extract_graph_data(json_result["json_ld"], hide_units_and_literals)
        else:
            nodes, links = json_result["nodes"], json_result["links"]
        
        # Return the successful result
        return {
            "success": True,
            "message": "PDF processed successfully",
            "knowledge_graph": json_result["json_ld"],
            "nodes": nodes,
            "links": links,
            "entities": entities,
            "relationships": relationships,
            "stage": "complete"
        }
import json
import re

def is_valid_json(json_string):
    """Check if a string is valid JSON."""
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

def extract_json_from_text(text):
    """Extract JSON from text that might contain extra content."""
    # Try to find JSON block between triple backticks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_pattern, text)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON in code blocks, try to find JSON with curly braces
    try:
        # Try to find the first { and last } that might enclose a JSON object
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_candidate = text[start_index:end_index+1]
            return json.loads(json_candidate)
    except json.JSONDecodeError:
        pass
    
    # If we still don't have valid JSON, clean the text and try again
    try:
        # Remove any non-JSON content by finding the first { and last }
        cleaned_text = re.sub(r'^[^{]*', '', text)
        cleaned_text = re.sub(r'[^}]*$', '', cleaned_text)
        return json.loads(cleaned_text)
    except (json.JSONDecodeError, IndexError):
        return None

def read_json_ld(data_string):
    """Parse JSON-LD data from string, with fallback to extraction if needed."""
    try:
        return json.loads(data_string)
    except json.JSONDecodeError:
        # Try to extract JSON from a text response
        json_data = extract_json_from_text(data_string)
        if json_data:
            return json_data
        raise ValueError("Failed to parse JSON-LD data")

def validate_knowledge_graph_json(json_data):
    """Validate that JSON has expected structure for a knowledge graph."""
    if not isinstance(json_data, dict):
        return False
    
    # Check for basic JSON-LD structure
    if "@context" not in json_data:
        return False
    
    if "@graph" not in json_data:
        return False
    
    if not isinstance(json_data["@graph"], list):
        return False
    
    # Check if graph has any nodes
    if len(json_data["@graph"]) == 0:
        return False
    
    # Check each node in the graph has an ID
    for node in json_data["@graph"]:
        if not isinstance(node, dict):
            return False
        if "@id" not in node:
            return False
    
    return True

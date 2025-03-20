import io
from PyPDF2 import PdfReader
import chardet
from langchain_text_splitters import NLTKTextSplitter
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def extract_text_from_pdf(pdf_file):
    """Extract text content from a PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        pdf_content = ""
        for page in pdf_reader.pages:
            pdf_content += page.extract_text() + "\n\n"
        return pdf_content
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")

def extract_text_from_txt(txt_file):
    """Extract text content from a TXT file."""
    try:
        # Read the content as bytes first to detect encoding
        content_bytes = txt_file.read()
        # Rewind the file pointer to the beginning
        txt_file.seek(0)
        
        # Detect encoding
        encoding_result = chardet.detect(content_bytes)
        encoding = encoding_result['encoding'] if encoding_result['encoding'] else 'utf-8'
        
        # Decode the content using the detected encoding
        content = content_bytes.decode(encoding)
        return content
    except Exception as e:
        raise Exception(f"Error extracting text from TXT file: {e}")

def chunk_text(pdf_content, api_choice="OpenAI API"):
    """Chunk text using simple newline splitting to avoid dependency issues."""
    # Adjust chunk size based on API provider
    # Groq has stricter token limits (6000 TPM), so we use much smaller chunks
    # The prompt templates are very large, so we need to leave room for them
    chunk_size = 500 if api_choice == "Groq API" else 3000
    
    # Manual chunking by paragraphs
    paragraphs = pdf_content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_in_chunks(text, max_length=4000, api_choice="OpenAI API"):
    """Split text into smaller chunks for processing."""
    # Adjust max length based on API provider
    if api_choice == "Groq API":
        max_length = 500  # Much smaller chunks for Groq due to large prompts
    
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind("\n")  # Split at the last newline
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:]
    chunks.append(text)
    return chunks

def read_prompt_file(prompt_file, default_prompt):
    """Read a prompt from a file or return the default prompt."""
    if prompt_file is not None:
        try:
            return prompt_file.read().decode("utf-8")
        except Exception as e:
            raise Exception(f"Error reading prompt file: {e}")
    return default_prompt

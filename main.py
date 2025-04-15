import os
import json
import tempfile
from typing import Dict, List, Any, Optional
import logging
from functools import lru_cache

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF processing
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAI integration
import openai
from openai import OpenAI
import tiktoken

# Environment variables and configuration
import dotenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF to JSON Processor",
    description="API for extracting structured data from PDFs and populating JSON templates",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client


@lru_cache()
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)

# Models for request/response


class ProcessingStatus(BaseModel):
    message: str
    task_id: Optional[str] = None


class JSONTemplate(BaseModel):
    template: Dict[str, Any]


# In-memory task storage (replace with a proper database in production)
processing_tasks = {}

# Helper functions


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise ValueError(f"Error extracting text from PDF: {e}")


def create_text_chunks(text: str, chunk_size: int = 4000, overlap: float = 0.1) -> List[str]:
    """Create chunks of text with overlap."""
    try:
        overlap_size = int(chunk_size * overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        logger.error(f"Error creating text chunks: {e}")
        raise ValueError(f"Error creating text chunks: {e}")


def get_token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Get the token count for a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def process_chunk_with_llm(
    chunk: str,
    template_fields: List[str],
    client: OpenAI,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Process a text chunk with LLM to extract structured data.
    """
    # Construct the prompt
    prompt = f"""
You are an expert data extraction system. Your task is to extract specific information from the text provided and format it according to the fields requested.

TEXT:
{chunk}

FIELDS TO EXTRACT:
{', '.join(template_fields)}

INSTRUCTIONS:
1. Carefully read the text and identify information related to each requested field.
2. If information for a field is not found in the text, set the value to null.
3. If information is found, extract it accurately and concisely.
4. Return ONLY a valid JSON object with the requested fields as keys.
5. Do not include explanations or additional text outside the JSON structure.

RESPONSE FORMAT:
{{
    "field1": "extracted value or null",
    "field2": "extracted value or null",
    ...
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        # Extract JSON from response
        result_text = response.choices[0].message.content
        result_data = json.loads(result_text)

        # Filter to include only the requested fields
        filtered_data = {field: result_data.get(
            field) for field in template_fields if field in result_data}

        return filtered_data
    except Exception as e:
        logger.error(f"Error processing chunk with LLM: {e}")
        raise ValueError(f"Error processing chunk with LLM: {e}")


def merge_results(results: List[Dict[str, Any]], template: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge extracted data from multiple chunks into the template JSON.
    Handles nested structures and different data types.
    """
    def merge_values(current_value: Any, new_value: Any) -> Any:
        # If both are dictionaries, merge them recursively
        if isinstance(current_value, dict) and isinstance(new_value, dict):
            for k, v in new_value.items():
                if k in current_value:
                    current_value[k] = merge_values(current_value[k], v)
                else:
                    current_value[k] = v
            return current_value

        # If both are lists, combine them
        elif isinstance(current_value, list) and isinstance(new_value, list):
            # Try to merge non-duplicate items
            return list(set(current_value + new_value))

        # If current value is None or empty, use new value
        elif current_value is None or current_value == "" or current_value == []:
            return new_value

        # Otherwise keep the current value (first occurrence takes precedence)
        return current_value

    # Create a deep copy of the template to avoid modifying the original
    result = json.loads(json.dumps(template))

    # Flatten the list of dictionaries into a single dictionary
    merged_data = {}
    for data_dict in results:
        for field, value in data_dict.items():
            if field in merged_data:
                merged_data[field] = merge_values(merged_data[field], value)
            else:
                merged_data[field] = value

    # Helper function to update nested fields
    def update_nested_field(target: Dict[str, Any], field_path: str, value: Any):
        parts = field_path.split('.')
        current = target

        # Navigate to the nested location
        for i, part in enumerate(parts[:-1]):
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]

        # Set the value at the leaf node
        if value is not None:
            current[parts[-1]] = value

    # Update the template with merged data
    for field, value in merged_data.items():
        update_nested_field(result, field, value)

    return result


async def process_pdf_background(
    pdf_path: str,
    template: Dict[str, Any],
    task_id: str,
    chunk_size: int = 4000,
    overlap_percentage: float = 0.1,
    model: str = "gpt-4o-mini"
):
    """Background processing task for PDF extraction and JSON population."""
    try:
        client = get_openai_client()

        # Extract text from PDF
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)

        # Create chunks with overlap
        logger.info(
            f"Creating text chunks with size {chunk_size} and overlap {overlap_percentage}")
        chunks = create_text_chunks(text, chunk_size, overlap_percentage)
        logger.info(f"Created {len(chunks)} chunks")

        # Flatten the template to get a list of all fields to extract
        def get_all_fields(json_obj, prefix="", result=None):
            if result is None:
                result = []

            if isinstance(json_obj, dict):
                for key, value in json_obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        get_all_fields(value, new_prefix, result)
                    else:
                        result.append(new_prefix)
            elif isinstance(json_obj, list) and json_obj and isinstance(json_obj[0], dict):
                get_all_fields(json_obj[0], f"{prefix}[0]", result)

            return result

        fields_to_extract = get_all_fields(template)

        # Process each chunk with LLM
        chunk_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            chunk_data = process_chunk_with_llm(
                chunk, fields_to_extract, client, model)
            chunk_results.append(chunk_data)

        # Merge results into the template
        final_json = merge_results(chunk_results, template)

        # Update task status
        processing_tasks[task_id] = {
            "status": "completed",
            "result": final_json
        }

        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        processing_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

        # Cleanup
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# API Endpoints


@app.post("/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    template_json: str = Query(..., description="JSON template string"),
    chunk_size: int = Query(
        4000, description="Maximum chunk size for processing"),
    overlap_percentage: float = Query(
        0.1, description="Overlap percentage between chunks (0.0-1.0)"),
    model: str = Query("gpt-4o-mini", description="OpenAI model to use")
):
    """
    Upload a PDF file and a JSON template for processing.

    - The PDF will be processed to extract text
    - Text will be chunked with the specified overlap
    - Each chunk will be processed with the LLM
    - Results will be merged into the provided JSON template
    """
    if file.filename is None or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Parse and validate template JSON
    try:
        template = json.loads(template_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON template format")

    # Create a unique task ID
    import uuid
    task_id = str(uuid.uuid4())

    # Save the uploaded PDF temporarily
    temp_pdf_path = f"temp_{task_id}.pdf"
    with open(temp_pdf_path, "wb") as pdf_file:
        pdf_file.write(await file.read())

    # Register task in memory
    processing_tasks[task_id] = {"status": "processing"}

    # Start the background processing task
    background_tasks.add_task(
        process_pdf_background,
        temp_pdf_path,
        template,
        task_id,
        chunk_size,
        overlap_percentage,
        model
    )

    return {"message": "PDF upload successful. Processing started.", "task_id": task_id}


@app.get("/process-status/{task_id}")
async def check_processing_status(task_id: str):
    """Check the status of a processing task."""
    if task_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = processing_tasks[task_id]
    if task_info["status"] == "completed":
        return {"status": "completed", "result": task_info["result"]}
    elif task_info["status"] == "failed":
        return {"status": "failed", "error": task_info.get("error", "Unknown error")}
    else:
        return {"status": "processing"}


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    template_json: str = Query(..., description="JSON template string"),
    chunk_size: int = Query(
        4000, description="Maximum chunk size for processing"),
    overlap_percentage: float = Query(
        0.1, description="Overlap percentage between chunks (0.0-1.0)"),
    model: str = Query("gpt-4o-mini", description="OpenAI model to use")
):
    """
    Synchronous endpoint for PDF processing (for smaller PDFs).
    This processes the PDF and returns the result immediately without background tasks.
    """
    if file.filename is None or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Parse and validate template JSON
    try:
        template = json.loads(template_json)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON template format")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        client = get_openai_client()

        # Extract text from PDF
        text = extract_text_from_pdf(temp_path)

        # Create chunks with overlap
        chunks = create_text_chunks(text, chunk_size, overlap_percentage)

        # Flatten the template to get a list of all fields to extract
        def get_all_fields(json_obj, prefix="", result=None):
            if result is None:
                result = []

            if isinstance(json_obj, dict):
                for key, value in json_obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, (dict, list)):
                        get_all_fields(value, new_prefix, result)
                    else:
                        result.append(new_prefix)
            elif isinstance(json_obj, list) and json_obj and isinstance(json_obj[0], dict):
                get_all_fields(json_obj[0], f"{prefix}[0]", result)

            return result

        fields_to_extract = get_all_fields(template)

        # Process each chunk with LLM
        chunk_results = []
        for chunk in chunks:
            chunk_data = process_chunk_with_llm(
                chunk, fields_to_extract, client, model)
            chunk_results.append(chunk_data)

        # Merge results into the template
        final_json = merge_results(chunk_results, template)

        return final_json

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing PDF: {str(e)}")

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Health check endpoint


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

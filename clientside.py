# Client-side example for using the API

import requests
import json
import time
import os

# API endpoint
API_URL = "http://localhost:8000"


def upload_pdf_for_processing(pdf_path, json_template):
    """
    Upload a PDF file for processing with a JSON template.

    Args:
        pdf_path (str): Path to the PDF file
        json_template (dict): JSON template to populate

    Returns:
        dict: API response including task_id
    """
    # Convert template to string
    template_str = json.dumps(json_template)

    # Prepare files and data
    files = {'file': open(pdf_path, 'rb')}
    params = {
        'template_json': template_str,
        'chunk_size': 4000,
        'overlap_percentage': 0.1,
        'model': 'gpt-4o-mini'
    }

    # Make the request
    response = requests.post(
        f"{API_URL}/upload-pdf",
        files=files,
        params=params
    )

    # Close the file
    files['file'].close()

    # Return the response
    return response.json()


def check_processing_status(task_id):
    """
    Check the status of a processing task.

    Args:
        task_id (str): Task ID returned from upload_pdf_for_processing

    Returns:
        dict: Status information
    """
    response = requests.get(f"{API_URL}/process-status/{task_id}")
    return response.json()


def process_pdf_sync(pdf_path, json_template):
    """
    Process a PDF synchronously (for smaller PDFs).

    Args:
        pdf_path (str): Path to the PDF file
        json_template (dict): JSON template to populate

    Returns:
        dict: Populated JSON template
    """
    # Convert template to string
    template_str = json.dumps(json_template)

    # Prepare files and data
    files = {'file': open(pdf_path, 'rb')}
    params = {
        'template_json': template_str,
        'chunk_size': 4000,
        'overlap_percentage': 0.1,
        'model': 'gpt-4o-mini'
    }

    # Make the request
    response = requests.post(
        f"{API_URL}/process-pdf",
        files=files,
        params=params
    )

    # Close the file
    files['file'].close()

    # Return the response
    return response.json()


# Example usage
if __name__ == "__main__":
    # Example JSON template
    template = {
        "document_info": {
            "title": None,
            "author": None,
            "creation_date": None,
            "document_type": None
        },
        "content_summary": None,
        "key_findings": [],
        "people_mentioned": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "financial_data": {
            "amounts": [],
            "currencies": [],
            "fiscal_years": []
        },
        "contact_information": {
            "emails": [],
            "phones": [],
            "addresses": []
        }
    }

    # Path to PDF file
    pdf_path = "example.pdf"

    # Choose either async or sync processing based on file size
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

    if file_size_mb < 10:  # For PDFs smaller than 10MB
        print("Using synchronous processing...")
        result = process_pdf_sync(pdf_path, template)
        print(json.dumps(result, indent=2))

    else:  # For larger PDFs
        print("Using asynchronous processing...")
        # Upload the PDF
        upload_response = upload_pdf_for_processing(pdf_path, template)
        task_id = upload_response["task_id"]
        print(f"Task ID: {task_id}")

        # Check status periodically until done
        while True:
            status = check_processing_status(task_id)
            print(f"Status: {status['status']}")

            if status['status'] == 'completed':
                print("Processing completed!")
                print(json.dumps(status['result'], indent=2))
                break
            elif status['status'] == 'failed':
                print(
                    f"Processing failed: {status.get('error', 'Unknown error')}")
                break

            # Wait a bit before checking again
            time.sleep(5)

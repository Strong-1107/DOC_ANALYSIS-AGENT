from datetime import datetime
import os
import time
import json
import tempfile
from typing import List, Dict
from openai import OpenAI, AssistantEventHandler
from openai.type.beta.threads.runs import ToolCallDeltaObject
from typing_extensions import override
from docx import Document


#Configuration
HOA_DOCS_DIR = "./input/hoa_document" #Paty to your HOA document
MODEL_NAME = "gpt-4o-mini" #Consider gpt-4-turbo-preview or gpt-4o for best balance of cost, speed, and token handling
ASSISTANT_NAME = "HOA Document Analyzer"
VECTOR_STORE_NAME = "HOA DOCUMENT"
TEMPERATURE = 0.1
MAX_RETRIES = 3 # Number of retries for API calls
RETRY_DELAY = 5
OUTPUT_DIR = "./output"

AUTHORITY_RANKING = {
    "CC&R Amendments": 1,
    "CC&Rs": 2,
    "Bylaws": 3,
    "Articles of Incorporation": 4,
    "Operating Rules": 5,
    "Election Rules": 6,
    "Annual Budget Report": 7,
    "Financial Statements": 8,
    "Reserve Study": 9,
    "Reserve Fund": 10,
    "Fine Schedule": 11,
    "Assessment Enforcement": 12,
    "Meeting Minutes": 13,
    "Additional Operational Policies & Guidelines": 14,
    "Insurance & Evidence of Insurance (COI)": 15,
    "Flood & General Liability Insurance": 16,
}

EXTRACTION_QUESTIONS = [
    "What is the official name of the homeowners association as indicated in the documents? (If multiple sources mention the name, use the information from the highest-ranked document.)",
    "What details are provided about the monthly dues (amounts, payment schedule, and any related conditions)? (Prioritize details from the highest-ranking file available, and note if the dues are aggregate or per property.)",
    "What information is available regarding fee increases and special assessments, including any criteria, frequency, or conditions under which they occur? (Reference the highest-priority document if multiple files address these topics.)",
    "How is the overall financial health of the HOA described, including any metrics, ratings, or commentary on fiscal stability? (Use details from the document highest in the ranking order when available.)",
    "What details are offered about the reserve fund (such as its balance, purpose, and allocation policies)? (If several documents provide this information, select details from the top-ranked source.)",
    "How is the HOA budget allocated among various expense categories, and what insights or breakdowns are provided? (Use the highest-authority source available.)",
    "What does the documentation reveal about the reputation of the management team (including performance, responsiveness, or community feedback)? (Reference the highest-priority document when multiple documents mention management reputation.)",
    "What issues or complaints have been documented, and what information is provided on how they were handled or resolved? (If details come from various sources, follow the ranking order to determine the authoritative source.)",
    "What specific rules and restrictions govern the community, and how are these policies structured or enforced? (Use the highest-ranked document addressing rules and restrictions.)",
    "What policies are in place regarding pets (e.g., permitted types, restrictions, approval processes, or limits)? (If multiple documents include pet policies, prioritize according to the given ranking order.)",
    "What information is provided about short-term rental policies, including any limitations or guidelines? (Refer to the highest-authority document if several files discuss this topic.)",
    "What details are included regarding capital improvements (such as planned projects, recent upgrades, or funding for improvements)? (Prioritize information from the document highest in the provided list.)",
    "How are the community amenities and overall property condition described in the documents? (Use the details from the top-ranked document available on amenities and conditions.)",
    "What information is available on the HOA’s governance practices and transparency, including decision-making processes and access to records? (If multiple documents offer insights, choose the details from the highest-authority file.)",
    "What enforcement measures and fine structures are documented for policy violations, and what are the associated procedures? (When conflicting information exists, refer to the highest-priority source such as Fine Schedule or Assessment Enforcement.)",
    "How does the HOA address routine maintenance and emergency situations, including any protocols or response plans? (Use the highest-ranked document that discusses maintenance and emergencies.)",
    "What processes are outlined for resolving disputes among residents or between residents and management? (If details are provided in several documents, prioritize using the ranking order.)",
    "What details are provided on insurance policies and service coverage, including scope, limitations, and any notable exclusions? (Use the highest-authority document among those addressing insurance, e.g., Insurance & Evidence of Insurance (COI) or Flood & General Liability Insurance.)",
    "What legal or regulatory issues have been identified, and how does the HOA address or mitigate these challenges? (Prioritize details from the highest-ranked document that discusses legal or regulatory matters.)",
    "What evidence or information is provided about resident engagement, involvement, or feedback within the community? (If multiple sources offer information on resident engagement, use the details from the highest-ranked document.)",
]

client = OpenAI

class EventHandler(AssistantEventHandler):
    def __init__(self):
        self.response_content = ""
        self.source_document = set()
    
    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nTool Called: {tool_call.type}", flush=True)

    @override
    def on_text_created(self, text):
        self.response_content += text.value + "\n"

    @override
    def on_file_citation_created(self, file_citation):
        try:
            file  = client.files.retrieve(file_citation.file_id)
            self.source_document.add(file.filename)
        except Exception as e:
            print(f"Error retrieving file citation: {e}")


    @override
    def on_message_done(self, message):
        print("\nMessage completed", flush=True)

def read_word_document(file_path: str) -> str:
    """Reads a word document and returns its text content."""
    try:
        document = Document(file_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as e:
        print(f"Error reading Word document {file_path}: {e}")
        return ""
    
def prepare_files(hoa_docs_dir: str) -> List[Dict[str, str]]:
    """Prepares a list of files with their content, handling different file type"""
    allowed_extensions = {'.doc', '.docx', '.pdf', '.txt', '.md'}
    file_paths = [
        os.path.join(hoa_docs_dir, filename)
        for filename in os.listdir(hoa_docs_dir)
        if os.path.isfile(os.path.join(hoa_docs_dir, filename))
        and not filename.startswith("~$")
        and os.path.splittext(filename)[1].lower() in allowed_extensions
    ]

    if not file_paths:
        print("No valid files with supported extensions found for upload")
        exit(1)
    
    files_with_content = []
    for file_path in file_paths:
        try:
            file_extension = os.path.splittext(file_path)[1].lower()
            if file_extension in ['.doc', '.docx']:
                content = read_word_document(file_path)
            elif file_extension == '.pdf':
                try:
                    from PyPDF2 import PdfReader
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        content = "".join(page.extract_txt() for page in reader.pages)
                except ImportError:
                    print("PyPDF2 is not installed. Please install it to process PDF files")
                    content = ""
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
                    content = ""
            else: # .txt, .md
                with open(file_path, "r", encoding = "utf-8") as f:
                    content = f.read()

            if content:
                files_with_content.append({"path": file_path, "content": content})
            else:
                print(f"Could not extract content from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return files_with_content

def create_or_update_assistant(client: OpenAI) -> any:
    """Creates a new Assistant or updates and existing one with File Search enabled."""
    try:
        #List existing assistants without order_by parameter
        assistants = client.beta.assistants.list()

        for asst in assistants.data:
            if asst.name == ASSISTANT_NAME:
                client.beta.assistants.delete(asst.id)
                print(f"Deleted existing assistant with ID: {asst.id}")

        # Create fresh assistant
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions=f"""
            You are an expert in HOA documents. Accuracy is exremely important.
            When answering, always extract inforation directly from the provided documents.
            If using file search, return the most relevant sections word-for-word and cite the document name.
            If no relevant information is foundk explicitly state: 'No relevant data found in the uploaded documents.'
            Do Not answer from general knowledge-only use the retrieved documents.

            Use this Authority  Ranking to prioritize information sources (1 is highest priority):
            {json.dumps(AUTHORITY_RANKING, indent=2)}

            When multiple documents containt relevant information, always prioritize information from the highest-ranked source.
            Include the source document name in your response.
            """,
            model = MODEL_NAME,
            tools=[{"type": "file_search"}],
            temperature=TEMPERATURE,
        )

        print(f"Created fresh assistant with ID: {assistant.id}")
        return assistant
    except Exception as e:
        print(f"Error creating/updating assistant: {e}")
        raise


def verify_assistant_setup(client: OpenAI, assistant_id: str, vector_store_id: str) -> bool:
    """Verifies the assistant is properly configured with the vector store."""
    assistant = client.beta.assistants.retrieve(assistant_id)

    if not hasattr(assistant, 'tool_resources') or \
       not assistant.tool_resources or \
       not assistant.tool_resources.file_search or \
       vector_store_id not in assistant.tool_resources.file_search.vector_store_ids:
        print ("Assistant not properly configured with vector store")
        return False
    print("Assistant properly configured with vector store")
    return True

def create_or_retrieve_vector_store(client: OpenAI) -> any:
    """Creates a new Vector Store or retrieves an existing one by name."""
    try:
        # Retrieve existing vector stores
        vector_stores = client.beta.vector_stores.list(order="desc", order_by="created_at")
        for vs in vector_stores.data:
            if vs.name == VECTOR_STORE_NAME:
                print(f"Found existing vector store with ID: {vs.id}")
                return vs
        # If no vector store with the specified name is found, create a new one
        raise ValueError(f"No vector store found with name: {VECTOR_STORE_NAME}")
    except Exception as e:
        print(f"An error occurred while typing to retireve the vetor store" {e})
        print("Creating a new vector store...")

        vector_store = client.beta.vector_stores.create(name=VECTOR_STORE_NAME)
        print(f"Vector store created with ID: {vector_store.id}")
        return vector_store


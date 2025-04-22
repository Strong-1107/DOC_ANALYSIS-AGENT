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
def uploaded_files_to_vector_store(client: OpenAI, vector_store_id: str, files_with_content: List[]) -> Doct:
    """Upload files to the vector store."""
    uploaded_file_ids = []
    fid_to_fpath = {}
    for file_data in files_with_content:
        try:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_data["path"])[1], delete=False) as temp_file:
                temp_file.write(file_data["content"].encode('utf-8'))
                temp_file_path = temp_file.name
            
            with open(temp_file_path, "rb") as file_stream:
                uploaded_file = client.files.create(file=file_stream, purpose="assistants")
                uploaded_file_ids.append(uploaded_file.id)
                fid_to_fpath[uploaded_file.id] = file_data['path']
                print(f"Uploaded file {file_data['path']} with ID: {uploaded_file.id}")
        except Exception as e:
            print(f"Error uploading {file_data['path']}: {e}")
        
        finally:
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
        if uploaded_file_ids:
            file_batch_attachment = client.beta.vector_stores.file_batched.create(
                vector_store_id = vector_store_id,
                file_ids=uploaded_file_ids
            )

            #Wait for file processing to complete
            while True:
                file_batch_attachment = client.beta.vector_stores.file_batches.retrieve(
                    vector_store_id=vector_store_id,
                    batch_id=file_batch_attachment.id
                )

                if file_batch_attachment.status == "completed":
                    print("File batch processing completed.")
                    break
                elif file_batch_attachment.status == "failed":
                    print("File batch processing failed")
                    exit(1)
                print("Waiting for file batch processing...")
                time.sleep(5)
            print(f"File batch explicitly added to vector store. Status: {file_batch_attachment.status}")
        else:
            print("No files were successfully uploaded.")
            exit(1)

        return fid_to_fpath
    def update_assistant(client: OpenAI, assistant_id: str, vector_store_id: str) -> None:
            """Updates the Assistant to use the Vector Store."""
            assistant = client.beta.assistants.update(
                assistant_id = assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            print("Assistant updated to use vector store.")

            assistant = client.beta.assistants.retrieve(assistant_id)
            print("\nAssistant Tool Resources:", assistant.tool_resources)

            if not assistant.tool_resources.file_search or vector_store_id not in assistant.tool_resources.file_search.vector_store_ids:
                print("Error: Assistant is NOT correctlyu linked to the vector store.")
                exit(1)
            else:
                print("Assistant is corectly linked to vector store:", assistant.tool_resources.file_search.vector_store_ids)
    def ask_question(client: OpenAI, assistant_id: str, question: str) -> Dict[str, str]:
        """Vector store-enabled question handler"""
        try:
            #create new thread
            thread = client.beta.threads.create()

            # Add the question
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"Search the documents and answer: {question}"
            )

            # Create run with file search enabled
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
                instructions="""
                Please address the used as Corbin. The user has a premium account.
                Use file search to find relevant information.
                Prioritize extracting information from the highest-ranked document according to the Authority Ranking.
                Provide both a detailed answer and a brief summary.
                Be extremely accurate.
                Format the response as:
                DETAILED ANSWER:
                [Your detailed response here]

                Summary:
                [Your brief summary here]
                """
            )

            # Wait for processing
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

                if run_status.status == 'completed':
                    #Get response with citations
                    messages = client.beta.threads.messages.list(
                        thread_id = thread.id,
                        order="desc",
                        limit=1
                    )

                    response = messages.data[0].content[0].text.value
                    sources = []
                    source_citations = []

                    # Extract file citations
                    for content in messages.data[0].content:
                        if hasattr(content, 'text'):
                           for annotation in content.text.annotations:
                               if annotation.type == "file_citation":
                                   file = client.files.retrieve(annotation.file_citation.file_id)
                                   sources.append(file.filename)
                                   source_citations.append(annotation.to_dict())

                    # Parse response sections
                    parts = response.split("SUMMARY:")
                    detailed = parts[0].replace("DETAILED ANSWER:", "").strip()
                    summary = parts[1].strip() if len(parts) > 1 else detailed[:200]

                    return {
                        "question": question,
                        "answer": detailed,
                        "summary": summary,
                        "source": ", ".join(set(sources)) if sources else "No specific sources cited",
                        "source_citations": source_citations
                    }
                if run_status in ['failed', 'cancelled', 'expired']:
                    return {
                        "question": question,
                        "answer": f"Search failed: {run_status.status}",
                        "summary": "Document search error",
                        "source": "N/A",
                        "source_citations": [],
                    } 

                time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            return {
                "question": question,
                "answer": str(e),
                "summary": "Processing error",
                "source": "N/A",
                "source_citations": []
            } 

    def ask_questions(client: OpenAI, assistant_id: str, questions: List[str]) -> List[Dict[str, str]]:
        """Sequential question processor"""
        results = []

        for idx, question in enumerate(questions, 1):
            print(f"\nProcessing question {idx}/{len(questions)}")
            result = ask_question(client, assistant_id, question)
            results.append(result)
            time.sleep(2)

        return results
    
    def create_summary_table(fid_to_fpath: Dict[str, str], responses: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Generates a JSON summary table from the responses, prioritizing by authority ranking."""
    summary_table = []
    categories = [
        "HOA Name",
          "Monthly Dues", "Fee Increases", "Financial Health", "Reserve Fund",
        "HOA Budget Allocation", "Management Reputation", "Documented Issues", "Community Rules", "Pet Policies",
        "Short-Term Rentals", "Capital Improvements", "Community Amenities", "Governance Practices", "Enforcement Measures",
        "Routine Maintenance", "Dispute Resolution", "Insurance Policies", "Legal Issues", "Resident Engagement"
    ]

    for i, category in enumerate(categories):
        response = next((r for r in responses if r["question"].startswith(EXTRACTION_QUESTIONS[i][:50])), None)

        if response:
            summary_table.append({
                "Category": category,
                "Findings": response["summary"] ,
                "Source": response["source_citations"]
            })
        else:
            summary_table.append({"Category": category, "Findings": "No information found.", "Source": "N/A"})

    return summary_table


def main():
    """Main function to orchestrate the process."""
    try:
        # 1. Prepare Files
        files_with_content = prepare_files(HOA_DOCS_DIR)

        # 2. Create or Retrieve Assistant and Vector Store
        assistant = create_or_update_assistant(client)
        vector_store = create_or_retrieve_vector_store(client)

        # 3. Upload and process files
        fid_to_fpath = upload_files_to_vector_store(client, vector_store.id, files_with_content)
        update_assistant(client, assistant.id, vector_store.id)
        
        # 4. Verify setup before proceeding
        if not verify_assistant_setup(client, assistant.id, vector_store.id):
            raise Exception("Assistant setup verification failed")
        
        # 5. Ask Questions
        responses = ask_questions(client, assistant.id, EXTRACTION_QUESTIONS)
        
        # 6. Create Summary Table
        summary_table = create_summary_table(fid_to_fpath, responses)
        
        # 7. Output JSON
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nSaving JSON Summary Table...")
        summary_fname = os.path.join(OUTPUT_DIR, "{}-summary.json".format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        with open(summary_fname, "w") as s_f:
            json.dump(summary_table, s_f, indent=4)
        print("\nSummary Table saved as:", summary_fname)

        print("\nSaving JSON Answer Table...")
        answers_fname = os.path.join(OUTPUT_DIR, "{}-answers.json".format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
        with open(answers_fname, "w") as a_f:
            json.dump(responses, a_f, indent=4)
        print("Answers saved as:", answers_fname)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting program.")

if __name__ == "__main__":
    main()

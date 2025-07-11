# Multi-Agent Legal Contract Analyzer

This project is a legal contract analysis application built with Streamlit, LlamaIndex, Ollama, and Qdrant. It uses a multi-agent workflow to perform a deep analysis of legal documents (specifically contracts), identifying logical inconsistencies, potential risks, and legal compliance issues by leveraging a Retrieval-Augmented Generation (RAG) pipeline against a vector database of legal texts.

The application is primarily designed to analyze contracts based on Vietnamese law, and its UI and prompts are in Vietnamese.

# Features

File Upload & Text Input: Analyze contracts by uploading PDF, DOCX, or TXT files, or by pasting raw text.

Multi-Agent Workflow: Deploys specialized agents for different analysis tasks:

Context Agent: Extracts metadata like contract type and cited laws.

Logic Agent: Detects internal contradictions and inconsistent terminology.

Risk Agent: Identifies one-sided clauses, vague terms, and missing standard clauses.

Legal Agent: Flags potential legal issues for verification.

RAG-Powered Legal Verification: Uses a Qdrant vector store to verify legal "red flags" against a knowledge base of laws and regulations.

Interactive Highlighting: Displays the analysis results by highlighting relevant sections of the contract directly in the UI.

Live Logging: Provides a real-time log of the workflow's progress during analysis.

Tech Stack & Architecture

Frontend: Streamlit

Core Logic & Orchestration: LlamaIndex Workflow (llama-index-utils-workflow)

LLM Serving: Ollama

Vector Database: Qdrant

Models:

LLM: qwen3:8b (or any other compatible model configured in .env)

Embedding Model: bkai-foundation-models/vietnamese-bi-encoder

# System Flow:

User provides a contract via the Streamlit UI.

The MultiAgentContractReviewWorkflow is triggered.

Agents analyze the contract in parallel (logic, risk, legal red flags).

The Legal Agent's findings are used to query the RAG pipeline.

The RAG pipeline, using a QueryEngineTool, retrieves relevant legal context from the Qdrant vector store.

The retrieved context is used to verify or provide information on the flagged legal issues.

A final agent synthesizes all findings into a comprehensive report.

The report and highlighted annotations are displayed in the Streamlit UI.

# Prerequisites

Before you begin, ensure you have the following installed:

Python 3.9+

Docker and Docker Compose (Recommended for running Qdrant)

Ollama installed and running. Ollama Installation Guide

# Setup & Installation

Follow these steps to get the project running locally.

1. Clone the Repository
git clone <your-repository-url>
cd multi-agent-contract-analyzer

2. Create a Virtual Environment and Install Dependencies

It's highly recommended to use a virtual environment.

# Create the virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

3. Set Up External Services (Ollama & Qdrant)

A. Ollama

Ensure the Ollama service is running. Then, pull the necessary models from the command line:

# Pull the main instruction-tuned LLM
ollama pull qwen3:8b

# Pull the Vietnamese embedding model
ollama pull bkai-foundation-models/vietnamese-bi-encoder

Note: If you wish to use different models, you can change them in the .env file later.

B. Qdrant

The easiest way to run Qdrant is with Docker.

docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

This command starts a Qdrant container, exposes the gRPC port (6333), and persists its data in a qdrant_storage directory in your project folder.

4. Configure Environment Variables

The application uses a .env file for configuration.

Create a file named .env in the root of the project directory.

Copy the contents of config.py as a template or use the example below, and adjust the values for your local setup.

5. Populate the Qdrant Vector Store

The application queries an existing Qdrant collection but does not include a data ingestion script. You must populate the collection specified in your .env file (law_collection_v6 by default) before the RAG functionality will work.

Gather your source documents: Place your legal documents (e.g., PDFs of Vietnamese laws) into a directory (e.g., data/).

Create an ingestion script.

Run the ingestion script.

python ingest.py

Important: The RAG filtering in workflow.py relies on specific metadata. For best results, your ingestion process should ideally add metadata to each node, such as law_normalized_simple and article_number. This requires a more advanced ingestion pipeline than the simple example above.

Running the Application

Once your setup is complete and your Qdrant collection is populated, you can run the Streamlit app:

streamlit run app.py

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

Connection Error to Ollama/Qdrant:

Ensure both Ollama and Qdrant services/containers are running.

Double-check the OLLAMA_BASE_URL, QDRANT_HOST, and QDRANT_PORT in your .env file.

Make sure there are no firewall rules blocking the connection.

qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 404... collection ... does not exist!

This means you have not populated the Qdrant collection yet. Run your ingest.py script.

Ensure the QDRANT_COLLECTION name in your .env file matches the one used in your ingestion script.

Slow Performance:

Analysis can be slow, especially when running large LLMs locally. The REQUEST_TIMEOUT has been set high to accommodate this.

Ensure you have sufficient RAM and, if possible, a dedicated GPU for Ollama.

Incorrect Highlighting:

The highlighting feature relies on matching exact quotes (verbatim) from the contract. If the LLM slightly alters the quote in its response, the highlight may fail. The prompts are designed to request verbatim quotes to minimize this.
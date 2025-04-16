# app.py
import streamlit as st
import asyncio
import logging
import time
from typing import List

# Import from project modules
import config
import utils
from workflow import (
    initialize_analysis_system,
    MultiAgentContractReviewWorkflow,
    WorkflowStartEvent,
    ProgressEvent,
    FinalOutputEvent,
    StopEvent,
    AnalysisResultsEvent, # Import intermediate events if needed for debugging
    ContextAnalyzedEvent
)

# --- Logging Configuration ---
# Make sure logs appear in the console where Streamlit runs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Page Config ---
st.set_page_config(page_title="Contract Analyzer", layout="wide")
st.title("📄 Multi-Agent Legal Contract Analyzer")
st.caption(f"Using LLM: `{config.LLM_MODEL}`, Embed: `{config.EMBED_MODEL}` | Ollama: `{config.OLLAMA_BASE_URL}` | Qdrant: `{config.QDRANT_HOST}:{config.QDRANT_PORT}` | Collection: `{config.QDRANT_COLLECTION}`")
st.markdown("---")

# --- Initialization (Cached) ---
# Use st.cache_resource to initialize the system only once
@st.cache_resource(show_spinner="Initializing Analysis System (LLM, VectorDB...). This may take a moment.")
def get_analysis_tool():
    # Need to run the async initialization function
    # asyncio.run() should work here as cache_resource runs it once outside the main script flow
    try:
        tool = asyncio.run(initialize_analysis_system())
        return tool
    except Exception as e:
         # If init fails, cache_resource might prevent retries easily.
         # Log the error and display it, stopping the app might be necessary.
         logger.error(f"CRITICAL: Failed to initialize analysis system in cache_resource: {e}", exc_info=True)
         st.error(f"Fatal Error: Could not initialize the analysis system. Please check logs. Error: {e}")
         st.stop()
         return None # Should not be reached if st.stop() works

law_tool = get_analysis_tool()

if not law_tool:
    # This case might occur if st.stop() didn't work as expected in the cache func
    st.error("Analysis tool could not be loaded. App cannot proceed.")
    st.stop()
else:
    st.sidebar.success("✅ Analysis System Initialized")
    st.sidebar.markdown(f"**RAG Tool:** `{law_tool.metadata.name}`")


# --- Session State Initialization ---
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'progress_updates' not in st.session_state:
    st.session_state.progress_updates = []
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# --- UI Elements ---
col1, col2 = st.columns([2, 3]) # Input column, Output column

with col1:
    st.subheader("Contract Input")
    input_method = st.radio("Input Method:", ("Upload File", "Paste Text"), horizontal=True, key="input_method")

    contract_text = ""
    uploaded_file = None
    text_input = ""

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Contract (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="file_uploader"
        )
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name
            try:
                with st.spinner(f"Extracting text from {file_name}..."):
                    contract_text = utils.extract_text(file_bytes, file_name)
                st.text_area("Extracted Text (Preview)", contract_text[:1000] + ("..." if len(contract_text) > 1000 else ""), height=150, disabled=True, key="text_preview")
            except Exception as e:
                st.error(f"Error extracting text: {e}")
                logger.error(f"Failed to extract text from {file_name}: {e}", exc_info=True)
                contract_text = "" # Reset text on error

    elif input_method == "Paste Text":
        text_input = st.text_area("Paste Contract Text Here:", height=400, key="text_input_area")
        if text_input:
            contract_text = text_input.strip()

    # Use contract_text if available from either method
    final_contract_text = contract_text if contract_text else ""

    analyze_button = st.button(
        "🔍 Analyze Contract",
        key="analyze_button",
        disabled=(not final_contract_text or st.session_state.analysis_running),
        use_container_width=True
    )

# --- Analysis Execution ---
if analyze_button and final_contract_text:
    st.session_state.analysis_running = True
    st.session_state.final_report = None # Clear previous report
    st.session_state.progress_updates = ["Starting analysis..."] # Initial progress
    st.session_state.error_message = None # Clear previous error
    st.rerun() # Rerun to show spinner and disable button

elif st.session_state.analysis_running:
    # This block runs when analysis is in progress (after the rerun)
    with col2:
        st.subheader("Analysis Results")
        progress_area = st.empty()
        results_area = st.empty()

        # Display progress updates
        with progress_area.container():
             st.info("Analysis in progress...")
             progress_bar = st.progress(0)
             progress_text = st.empty()
             # Simple progress simulation (replace with actual if possible)
             max_steps = 4 # Preprocess, Context, Detail, Synthesize
             for i, update in enumerate(st.session_state.progress_updates):
                 progress_val = min(1.0, (i + 1) / max_steps)
                 progress_bar.progress(progress_val)
                 progress_text.text(f"Status: {update}")
             # Keep spinner while waiting for the async task
             st.spinner("Working...")


    # Define the callback function for progress
    # It needs to update session state and trigger a rerun
    def streamlit_progress_callback(event: ProgressEvent):
        logger.info(f"Workflow Progress: {event.progress}")
        if event.progress not in st.session_state.progress_updates:
             st.session_state.progress_updates.append(event.progress)
             # Need to trigger rerun carefully, might cause issues if too frequent
             # For now, let the main loop handle reruns based on state change


    async def run_analysis_async_wrapper():
        try:
            # Instantiate workflow with the callback
            workflow = MultiAgentContractReviewWorkflow(
                timeout=config.WORKFLOW_TIMEOUT,
                verbose=config.VERBOSE_WORKFLOW,
                progress_callback=streamlit_progress_callback # Pass the callback
            )

            # start_event = WorkflowStartEvent(
            #     contract_text=final_contract_text, # Use the text extracted earlier
            #     query=config.DEFAULT_QUERY,
            #     tools=[law_tool]
            # )

            # # Use run_event to get the final event
            # result_event = await workflow.run(**start_event)

            # if isinstance(result_event, StopEvent) and isinstance(result_event.result, FinalOutputEvent):
            #     st.session_state.final_report = result_event.result.report
            #     st.session_state.error_message = None
            #     logger.info("Workflow completed successfully.")
            # else:
            #     # Handle unexpected end or intermediate events if run_event returns them on error
            #     error_info = f"Workflow finished unexpectedly. Last event type: {type(result_event)}. Check logs for details."
            #     if hasattr(result_event, 'result') and result_event.result:
            #          error_info += f" Result data: {result_event.result}"
            #     st.session_state.error_message = error_info
            #     st.session_state.final_report = None
            #     logger.error(error_info)
            # 1. Define the data dictionary
            start_event_data = {
                "contract_text": final_contract_text, # Use the text extracted earlier
                "query": config.DEFAULT_QUERY,
                "tools": [law_tool] # Make sure law_tool is correctly initialized and passed as a list
            }

            # 2. Call workflow.run() UNPACKING the dictionary
            final_result = await workflow.run(**start_event_data)
            # ---- END CORRECTED PART ----

            # Check the type of the *returned result*, which should be FinalOutputEvent
            if isinstance(final_result, FinalOutputEvent):
                st.session_state.final_report = final_result.report
                st.session_state.error_message = None
                logger.info("Workflow completed successfully.")

        except Exception as e:
            error_msg = f"An error occurred during the workflow execution: {e}"
            logger.error(error_msg, exc_info=True)
            st.session_state.error_message = error_msg
            st.session_state.final_report = None
        finally:
            st.session_state.analysis_running = False
            # Trigger a final rerun to display results/errors
            st.rerun()

    # Run the async function using asyncio.run()
    # This blocks the Streamlit execution thread until the async task is complete
    # which is usually okay for a single long-running task initiated by user action.
    asyncio.run(run_analysis_async_wrapper())

# --- Display Results or Errors ---
with col2:
    # This part runs *after* the analysis is complete or errored out
    # because st.session_state.analysis_running is now False
    if not st.session_state.analysis_running:
        if st.session_state.error_message:
            st.error("Analysis Failed!")
            st.error(st.session_state.error_message)
        elif st.session_state.final_report:
            st.subheader("Analysis Results")
            st.success("Analysis Complete!")
            st.markdown(st.session_state.final_report)
        elif analyze_button: # If button was clicked but no report/error yet (shouldn't happen ideally)
             st.info("Analysis was triggered. Waiting for results...")
# app.py
import streamlit as st
import asyncio
import logging
import html # Standard Python HTML library
from logging import Handler, LogRecord
from typing import Optional, Tuple
# Import from project modules
import config
import utils
from workflow import (
    initialize_analysis_system,
    MultiAgentContractReviewWorkflow,
    WorkflowStartEvent, # Use correct start event
    ProgressEvent,
    FinalOutputEvent,
    # AnalysisResultsEvent, # Not directly used by UI
    # ContextAnalyzedEvent # Not directly used by UI
)
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import BaseTool 
# --- Custom Logging Handler ---
class StreamlitLogHandler(Handler):
    """Logs messages to a list in Streamlit session state."""
    def __init__(self, log_list_key="log_messages"):
        super().__init__()
        self.log_list_key = log_list_key
        # Initialize in session state if not present
        if self.log_list_key not in st.session_state:
            st.session_state[self.log_list_key] = []

    def emit(self, record: LogRecord):
        try:
            msg = self.format(record)
            # Ensure the key exists and is a list before appending
            if isinstance(st.session_state.get(self.log_list_key), list):
                st.session_state[self.log_list_key].append(msg + "\n")
            else:
                # If it doesn't exist or isn't a list, reset it
                st.session_state[self.log_list_key] = [msg + "\n"]
        except Exception:
            self.handleError(record)

# --- Logging Configuration ---
# Configure root logger first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
# Get specific loggers
logger = logging.getLogger(__name__) # Logger for the app itself
workflow_logger = logging.getLogger("WorkflowSteps") # Logger for workflow progress/steps
workflow_logger.setLevel(logging.INFO) # Set level for workflow logger
# Add Streamlit handler to workflow logger
# Ensure handler is not added multiple times on reruns
handler_exists = any(isinstance(h, StreamlitLogHandler) for h in workflow_logger.handlers)
if not handler_exists:
    log_handler_instance = StreamlitLogHandler(log_list_key='log_messages')
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    log_handler_instance.setFormatter(log_formatter)
    workflow_logger.addHandler(log_handler_instance)
    logger.info("Streamlit log handler added to workflow logger.")
else:
    logger.info("Streamlit log handler already attached to workflow logger.")

# --- Page Config ---
st.set_page_config(page_title="Contract Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Multi-Agent Legal Contract Analyzer")
st.caption(f"LLM: {config.LLM_MODEL} | Embed: {config.EMBED_MODEL} | RAG: {config.QDRANT_COLLECTION} | Timeout: {config.WORKFLOW_TIMEOUT}s")
st.markdown("---")

# --- Analysis System Initialization ---
@st.cache_resource(show_spinner="Initializing Analysis System...")
def get_analysis_resources() -> Optional[Tuple[VectorStoreIndex, BaseTool]]: # <<< Cập nhật tên hàm và type hint
    """Initializes the RAG tool and Index needed by the workflow."""
    try:
        # initialize_analysis_system trả về (index, law_tool)
        resources = asyncio.run(initialize_analysis_system())
        if resources is None:
            raise RuntimeError("initialize_analysis_system returned None, indicating failure.")
        return resources # Trả về tuple (index, law_tool)
    except Exception as e:
        logger.error(f"CRITICAL: Failed initialization: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize system. Check logs/services.")
        st.exception(e)
        st.stop()
        return None

# Get the initialized resources
analysis_resources = get_analysis_resources()
my_index: Optional[VectorStoreIndex] = None
law_tool: Optional[BaseTool] = None

if analysis_resources:
    my_index, law_tool = analysis_resources # <<< Giải nén tuple
    st.sidebar.success("✅ Analysis System Initialized")
    if law_tool and hasattr(law_tool, 'metadata'): # Kiểm tra trước khi truy cập metadata
        st.sidebar.markdown(f"**RAG Tool:** `{law_tool.metadata.name}`")
    else:
        st.sidebar.markdown("**RAG Tool:** `Metadata not available`")
    # st.sidebar.markdown(f"**Index Type:** `{type(my_index)}`") # Có thể thêm để debug
else:
    st.sidebar.error("❌ Analysis System Initialization Failed")
    st.stop()

# --- Session State Initialization ---
# Use .get() with default values for safer access
st.session_state.setdefault('analysis_running', False)
st.session_state.setdefault('progress_updates', [])
st.session_state.setdefault('final_report', None)
st.session_state.setdefault('error_message', None)
st.session_state.setdefault('log_messages', [])
st.session_state.setdefault('original_contract_text', "")
st.session_state.setdefault('annotations_data', []) # Store annotations data
st.markdown("""
<style>
/* Base highlight style */
.highlight {
    padding: 0.2em 0.4em;
    border-radius: 6px;
    position: relative;
    cursor: help;
    transition: background-color 0.2s ease;
    border-bottom: 2px dotted rgba(0,0,0,0.3);
    white-space: pre-wrap;
}

/* Highlight type coloring */
.highlight-legal {
    background-color: rgba(255, 230, 0, 0.3);
    border-color: #FFC107;
}
.highlight-logic {
    background-color: rgba(0, 188, 212, 0.25);
    border-color: #00BCD4;
}
.highlight-risk {
    background-color: rgba(244, 67, 54, 0.25);
    border-color: #F44336;
}

/* Status modifiers */
.status-citation-incorrect-confirmed {
    font-weight: bold;
}
.status-missing-mandatory-confirmed {
    font-weight: bold;
}
.status-citation-needs-review,
.status-contradiction-check-needed {
    border-style: dashed;
}

/* Tooltip styling using title attribute */
.highlight[title]:hover::after {
    content: attr(title);
    white-space: pre-wrap;
    position: absolute;
    bottom: 100%;
    left: 0;
    z-index: 10;
    background-color: #333;
    color: #fff;
    padding: 6px 10px;
    font-size: 12px;
    max-width: 300px;
    border-radius: 4px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    transform: translateY(-4px);
}

/* Container */
.contract-container {
    padding: 1em;
    border: 1px solid #ddd;
    background-color: #fdfdfd;
    font-family: 'Courier New', monospace;
    font-size: 15px;
    line-height: 1.6;
    color: #333;
    white-space: pre-wrap;
    word-wrap: break-word;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Optional scrollbar styling (Chrome/Edge) */
.contract-container::-webkit-scrollbar {
    width: 8px;
}
.contract-container::-webkit-scrollbar-thumb {
    background-color: #ccc;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)



# --- UI Layout ---
col1, col2 = st.columns([6, 4]) # Give more space to contract view
analyze_button_state = None

# --- State 1: Input ---
if not st.session_state.analysis_running and not st.session_state.final_report and not st.session_state.error_message:
    logger.info("Rendering Input State.")
    with col1:
        st.subheader("Nhập Hợp đồng")
        input_method = st.radio("Phương thức nhập:", ("Tải file lên", "Dán văn bản"), horizontal=True, key="input_method")
        contract_text_input = ""
        uploaded_file = None
        if input_method == "Tải file lên":
            uploaded_file = st.file_uploader("Tải hợp đồng (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="file_uploader")
            if uploaded_file:
                file_bytes = uploaded_file.getvalue()
                file_name = uploaded_file.name
                try:
                    with st.spinner(f"Đang trích xuất văn bản từ {file_name}..."):
                        contract_text_input = utils.extract_text(file_bytes, file_name)
                    logger.info(f"Trích xuất thành công từ {file_name}.")
                    st.text_area("Xem trước nội dung:", contract_text_input[:1000] + ("..." if len(contract_text_input) > 1000 else ""), height=150, disabled=True, key="text_preview")
                    st.session_state.original_contract_text = contract_text_input # Store full text
                except ValueError as ve:
                    st.error(f"Lỗi file: {ve}")
                    logger.error(f"File extraction error: {ve}", exc_info=True)
                    st.session_state.original_contract_text = ""
                except Exception as e:
                    st.error(f"Lỗi không xác định khi xử lý file: {e}")
                    logger.error(f"Unexpected file processing error: {e}", exc_info=True)
                    st.session_state.original_contract_text = ""
        elif input_method == "Dán văn bản":
            text_input_area = st.text_area("Dán nội dung hợp đồng vào đây:", height=400, key="text_input_area")
            if text_input_area:
                contract_text_input = text_input_area.strip()
                st.session_state.original_contract_text = contract_text_input

        analyze_button_state = st.button("🔍 Phân tích Hợp đồng", key="analyze_button", disabled=(not st.session_state.original_contract_text), use_container_width=True, type="primary")

    with col2:
        st.markdown("### Hướng dẫn")
        st.markdown("""
            1.  **Tải file** hợp đồng (PDF, DOCX, TXT) hoặc **Dán văn bản** vào ô bên trái.
            2.  Nhấn nút **Phân tích Hợp đồng**.
            3.  Chờ quá trình phân tích hoàn tất (có thể mất vài phút).
            4.  Xem kết quả:
                *   **Cột trái:** Văn bản hợp đồng với các điểm cần lưu ý được **highlight**. Di chuột vào phần highlight để xem chi tiết.
                *   **Cột phải:** Báo cáo phân tích chi tiết.
            """)
        st.info("Lưu ý: Kết quả phân tích chỉ mang tính tham khảo, cần có sự tư vấn của chuyên gia pháp lý.")

# --- ACTION: Handle Button Click ---
if analyze_button_state:
    logger.info("Analyze button clicked.")
    if st.session_state.original_contract_text:
        # Reset state for a new analysis
        st.session_state.analysis_running = True
        st.session_state.final_report = None
        st.session_state.error_message = None
        st.session_state.progress_updates = ["Bắt đầu phân tích..."]
        st.session_state.log_messages = [] # Clear previous logs
        st.session_state.annotations_data = [] # Clear previous annotations

        logger.info("Set analysis_running=True. Triggering rerun for analysis phase.")
        st.rerun() # Rerun to enter the 'Analysis Running' state
    else:
        st.warning("Vui lòng cung cấp nội dung hợp đồng trước khi phân tích.")
        logger.warning("Analyze clicked but no contract text found.")

# --- State 2: Analysis Running ---
elif st.session_state.analysis_running:
    logger.info("Rendering 'Analysis Running' state.")
    # Display progress updates
    with col1:
        st.info("⏳ Quá trình phân tích đang diễn ra...")
        st.write("Vui lòng chờ trong giây lát.")
    with col2:
        st.subheader("Trạng thái Phân tích")
        progress_area = st.empty()
        with progress_area.container():
            # Simplified progress display
            current_step_count = len(st.session_state.progress_updates)
            # Estimate total steps (adjust based on typical workflow events)
            total_steps_estimate = 5 # Preprocess, Context, Detail(LLM), Detail(RAG), Synthesize
            progress_value = min(1.0, (current_step_count) / total_steps_estimate)
            st.progress(progress_value)
            last_progress = st.session_state.progress_updates[-1] if st.session_state.progress_updates else "Khởi tạo..."
            st.text(f"Trạng thái: {last_progress}")
            st.spinner("Đang xử lý...")

    # Define the async wrapper to run the workflow
    async def run_analysis_async_wrapper():
        logger.info("Entering run_analysis_async_wrapper")
        active_log_handler = None
        for h in workflow_logger.handlers:
            if isinstance(h, StreamlitLogHandler):
                active_log_handler = h
                break
        if not active_log_handler:
            logger.error("Streamlit log handler instance not found!")
            # Proceed without handler if necessary, but log the error
        else:
            logger.info("Streamlit log handler found and active.")


        try:
            # Define callback within the async function's scope
            def streamlit_progress_callback(event: ProgressEvent):
                logger.info(f"Progress Callback Received: {event.progress}")
                if event.progress not in st.session_state.progress_updates:
                    st.session_state.progress_updates.append(event.progress)
                    # Trigger UI update from callback if possible/needed
                    # Note: Direct st.rerun() from async thread might be problematic
                    # Consider using st.session_state changes to trigger rerun on main thread check
                    logger.debug(f"Appended progress: {event.progress}. Rerunning UI.")
                    try:
                        # Need to be careful with reruns inside async callbacks
                        pass # Avoid direct rerun, rely on state change
                    except Exception as e:
                        logger.warning(f"Ignoring rerun error in callback: {e}")


            workflow = MultiAgentContractReviewWorkflow(
                timeout=config.WORKFLOW_TIMEOUT,
                verbose=config.VERBOSE_WORKFLOW,
                progress_callback=streamlit_progress_callback # Pass the callback
            )
            start_event = WorkflowStartEvent(
                contract_text=st.session_state.original_contract_text,
                query=config.DEFAULT_QUERY,
                tools=[law_tool], # Pass the initialized tool
                index=my_index 
            )
            logger.info("Starting workflow.run...")
            # result_event = await workflow.run_event(ev=start_event) # Use run_event
            result_output = await workflow.run(**start_event.model_dump()) # Sử dụng run() và truyền các tham số từ event
            logger.info("Workflow run finished.")

            if isinstance(result_output, FinalOutputEvent):
                final_output : FinalOutputEvent = result_output
                st.session_state.final_report = final_output.report
                st.session_state.annotations_data = final_output.annotations # Store annotations
                st.session_state.error_message = None
                logger.info("Workflow completed successfully.")
                logger.info(f"Report length: {len(final_output.report)}, Annotations count: {len(final_output.annotations)}")
            else:
                # Nếu run() không trả về FinalOutputEvent như mong đợi
                error_info = f"Workflow finished with unexpected result type: {type(result_output)}"
                logger.error(error_info)
                st.session_state.error_message = error_info
                st.session_state.final_report = None
                st.session_state.annotations_data = []

        except Exception as e:
            error_msg = f"Lỗi thực thi Workflow: {e}"
            logger.error(error_msg, exc_info=True)
            st.session_state.error_message = error_msg
            st.session_state.final_report = None
            st.session_state.annotations_data = []
        finally:
            st.session_state.analysis_running = False
            # No need to remove handler here if it's managed globally or per session
            logger.info("Analysis finished. Triggering rerun for final view.")
            # Need to trigger a rerun for the UI to update after async task finishes
            # Use st.experimental_rerun() or simply let Streamlit's loop handle it
            # after the state change (analysis_running=False)
            st.rerun() # Force rerun to show results/error

    # Run the async function
    asyncio.run(run_analysis_async_wrapper())
    # The script execution stops here in Streamlit while waiting for async to complete
    # Rerun is triggered inside the async wrapper's finally block


# --- State 3: Final View ---
elif not st.session_state.analysis_running and (st.session_state.final_report or st.session_state.error_message):
    logger.info("Rendering Final State (Results or Error).")
    original_text = st.session_state.get('original_contract_text', "")
    report_text = st.session_state.get('final_report', "")
    error_message = st.session_state.get('error_message', "")
    annotations_data = st.session_state.get('annotations_data', [])

    if error_message:
        logger.error(f"Displaying error message: {error_message}")
        with col1:
            st.error(f"**Lỗi Phân tích!**")
            st.error(error_message)
            # Show original text if available
            if original_text:
                st.subheader("Nội dung Hợp đồng Gốc")
                st.text_area("Nội dung gốc:", original_text, height=400, disabled=True, key="error_original_text")
        with col2:
            st.warning("Xem chi tiết lỗi ở cột bên trái.")
    elif report_text:
        logger.info(f"Displaying report. Report length: {len(report_text)}, Annotations: {len(annotations_data)}")
        # Highlight contract using annotations
        try:
            with st.spinner("Đang xử lý highlight..."):
                highlighted_contract_html = utils.highlight_contract_with_annotations(original_text, annotations_data)
                logger.info("Highlighting function completed.")
        except Exception as high_err:
            logger.error(f"Error during highlighting: {high_err}", exc_info=True)
            st.error(f"Lỗi khi tạo highlight: {high_err}")
            # Fallback to showing unhighlighted text
            highlighted_contract_html = html.escape(original_text).replace("\n", "<br>")

        with col1:
            st.subheader("Văn bản Hợp đồng (Highlight)")
            # Use the custom container class for styling and scroll
            # st_html(f'<div class="contract-container">{highlighted_contract_html}</div>', height=800, scrolling=True)
            st.markdown(f'<div class="contract-container">{highlighted_contract_html}</div>', unsafe_allow_html=True)

        with col2:
            st.subheader("Báo cáo Phân tích")
            # print(report_text) # Print the report text
            st.markdown(report_text, unsafe_allow_html=False) # Display report as Markdown

    # Expander for logs (always show if logs exist)
    log_messages = st.session_state.get('log_messages', [])
    if log_messages:
        st.markdown("---")
        with st.expander("Xem Log Phân tích Chi tiết", expanded=False):
            log_content = "".join(log_messages)
            st.code(log_content, language='log')

    # Add a button to start a new analysis
    if st.button("Phân tích Hợp đồng Khác", key="new_analysis_button"):
        # Reset relevant session state variables
        st.session_state.analysis_running = False
        st.session_state.final_report = None
        st.session_state.error_message = None
        st.session_state.progress_updates = []
        st.session_state.log_messages = []
        st.session_state.original_contract_text = ""
        st.session_state.annotations_data = []
        # Clear potentially cached file uploader state if needed (might not be necessary)
        # if 'file_uploader' in st.session_state: del st.session_state['file_uploader']
        logger.info("Resetting state for new analysis.")
        st.rerun()

# --- Fallback State (Should not be reached ideally) ---
else:
    logger.debug("Rendering initial/fallback state (no analysis running, no results/error).")
    # This state might be hit briefly on initial load or if state is inconsistent
    # It could simply pass or display a minimal placeholder if needed.
    # For now, the input state logic at the top handles the initial view.
    pass
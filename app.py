# app.py (Contract Analyzer - Modified for log deduplication)
import streamlit as st
import asyncio
import logging
import html 
from logging import Handler, LogRecord
from typing import Optional, Tuple, List

# Import from project modules
import config
import utils
from workflow import (
    initialize_analysis_system,
    MultiAgentContractReviewWorkflow,
    WorkflowStartEvent,
    ProgressEvent,
    FinalOutputEvent,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import BaseTool

# --- Custom Logging Handler ---
class StreamlitLogHandler(Handler):
    """Logs messages to a list in Streamlit session state AND updates an st.empty placeholder."""
    def __init__(self, log_list_key="log_messages", log_placeholder_key="log_placeholder_widget"):
        super().__init__()
        self.log_list_key = log_list_key
        self.log_placeholder_key = log_placeholder_key

        if self.log_list_key not in st.session_state or not isinstance(st.session_state.get(self.log_list_key), list):
            st.session_state[self.log_list_key] = []
        
    def format_log_html(self, messages: List[str]) -> str:
        escaped_messages = [html.escape(msg.strip()) for msg in messages]
        # Make sure each log message is on a new line in the HTML
        return f"<div class='progress-container-style'>{''.join(f'{m}<br>' for m in escaped_messages)}</div>"

    def emit(self, record: LogRecord):
        try:
            msg_formatted = self.format(record) # This already includes formatter's work (asctime, levelname, message)
            
            # Append to session state list (for persistence and final display)
            # Only append if not already present to prevent duplicates from handler re-emission within same event
            current_logs_list = st.session_state.get(self.log_list_key)
            if not isinstance(current_logs_list, list):
                current_logs_list = [] # Should have been initialized, but safeguard
                st.session_state[self.log_list_key] = current_logs_list
            
            # Add newline if not already there, for the list storage
            msg_to_store = msg_formatted + ("\n" if not msg_formatted.endswith("\n") else "")

            # Simple check to avoid exact duplicate consecutive messages in the list
            # This is a basic deduplication at the handler level for the stored list.
            if not current_logs_list or current_logs_list[-1].strip() != msg_formatted.strip():
                 current_logs_list.append(msg_to_store)
            # else:
            #     # Optionally log that a duplicate was suppressed from the list
            #     # print(f"DEBUG: Suppressed duplicate log message in list: {msg_formatted.strip()}")
            #     pass


            log_placeholder_widget = st.session_state.get(self.log_placeholder_key)
            if log_placeholder_widget:
                # For display, always re-render the complete list from session_state
                # This ensures the displayed log is always in sync with the stored log.
                full_log_content_html = self.format_log_html(st.session_state[self.log_list_key])
                log_placeholder_widget.markdown(full_log_content_html, unsafe_allow_html=True)

        except Exception as e:
            import sys
            sys.stderr.write(f"Error in StreamlitLogHandler.emit: {e}\n")
            self.handleError(record)

# --- Logging Configuration ---
# Define keys for session state
LOG_LIST_KEY = 'log_messages_contract_analyzer'
LOG_PLACEHOLDER_WIDGET_KEY = 'log_placeholder_contract_analyzer_widget'
LOG_HANDLER_INSTANCE_KEY = 'log_handler_instance_contract_analyzer'

# Configure root logger (basic setup, can be adjusted)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# Get app's own logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # App logger level

# Get workflow logger
workflow_logger = logging.getLogger("WorkflowSteps")
workflow_logger.setLevel(logging.INFO) # Workflow logger level
workflow_logger.propagate = False      # CRUCIAL: Prevent messages from going to root logger's handlers

# --- Singleton Handler Setup ---
if LOG_HANDLER_INSTANCE_KEY not in st.session_state:
    # Create and store the handler instance only once per session
    log_handler_instance = StreamlitLogHandler(
        log_list_key=LOG_LIST_KEY,
        log_placeholder_key=LOG_PLACEHOLDER_WIDGET_KEY
    )
    log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    log_handler_instance.setFormatter(log_formatter)
    st.session_state[LOG_HANDLER_INSTANCE_KEY] = log_handler_instance
    logger.info("NEW StreamlitLogHandler instance created and stored in session_state.")
else:
    # Retrieve the existing handler instance from session_state
    log_handler_instance = st.session_state[LOG_HANDLER_INSTANCE_KEY]
    logger.info("REUSED StreamlitLogHandler instance from session_state.")

# Ensure the workflow_logger has OUR handler and ONLY our handler of this type
# Remove any old StreamlitLogHandler instances first to be safe
for h in list(workflow_logger.handlers): # Iterate over a copy for safe removal
    if isinstance(h, StreamlitLogHandler):
        logger.info(f"Removing existing StreamlitLogHandler: {h} from workflow_logger.")
        workflow_logger.removeHandler(h)

# Add our singleton handler if it's not already there
if log_handler_instance not in workflow_logger.handlers:
    workflow_logger.addHandler(log_handler_instance)
    logger.info("Singleton StreamlitLogHandler added to workflow_logger.")
else:
    logger.info("Singleton StreamlitLogHandler already present in workflow_logger.handlers.")


# --- Page Config (Unchanged) ---
st.set_page_config(page_title="Contract Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("📄 Multi-Agent Legal Contract Analyzer")
st.caption(f"LLM: {config.LLM_MODEL} | Embed: {config.EMBED_MODEL} | RAG: {config.QDRANT_COLLECTION} | Timeout: {config.WORKFLOW_TIMEOUT}s")

# --- Analysis System Initialization (Unchanged) ---
@st.cache_resource(show_spinner="Initializing Analysis System...")
def get_analysis_resources() -> Optional[Tuple[VectorStoreIndex, BaseTool]]:
    try:
        resources = asyncio.run(initialize_analysis_system())
        if resources is None:
            raise RuntimeError("initialize_analysis_system returned None, indicating failure.")
        return resources
    except Exception as e:
        logger.error(f"CRITICAL: Failed initialization: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize system. Check logs/services.")
        st.exception(e)
        st.stop()

analysis_resources = get_analysis_resources()
my_index: Optional[VectorStoreIndex] = None
law_tool: Optional[BaseTool] = None

if analysis_resources:
    my_index, law_tool = analysis_resources
    st.sidebar.success("✅ Analysis System Initialized")
    if law_tool and hasattr(law_tool, 'metadata'):
        st.sidebar.markdown(f"**RAG Tool:** `{law_tool.metadata.name}`")
    else:
        st.sidebar.markdown("**RAG Tool:** `Metadata not available`")
else:
    st.sidebar.error("❌ Analysis System Initialization Failed")
    st.stop()

# --- Session State Initialization (Ensure keys are present) ---
st.session_state.setdefault('analysis_running', False)
st.session_state.setdefault('progress_updates', [])
st.session_state.setdefault('final_report', None)
st.session_state.setdefault('error_message', None)
st.session_state.setdefault(LOG_LIST_KEY, []) 
st.session_state.setdefault('original_contract_text', "")
st.session_state.setdefault('annotations_data', [])
st.session_state.setdefault('async_task_started_for_current_analysis', False)
st.session_state.setdefault(LOG_PLACEHOLDER_WIDGET_KEY, None) 

# --- Styles and Markdown (Unchanged from your last version with progress-container-style) ---
st.markdown("""
<style>
/* Styles from your original contract analyzer app */
.highlight { padding: 0.2em 0.4em; border-radius: 6px; position: relative; cursor: help; transition: background-color 0.2s ease; border-bottom: 2px dotted rgba(0,0,0,0.3); white-space: pre-wrap; }
.highlight-legal { background-color: rgba(255, 230, 0, 0.3); border-color: #FFC107; }
.highlight-logic { background-color: rgba(0, 188, 212, 0.25); border-color: #00BCD4; }
.highlight-risk { background-color: rgba(244, 67, 54, 0.25); border-color: #F44336; }
.status-citation-incorrect-confirmed { font-weight: bold; }
.status-missing-mandatory-confirmed { font-weight: bold; }
.status-citation-needs-review, .status-contradiction-check-needed { border-style: dashed; }
.highlight[title]:hover::after { content: attr(title); white-space: pre-wrap; position: absolute; bottom: 100%; left: 0; z-index: 10; background-color: #333; color: #fff; padding: 6px 10px; font-size: 12px; max-width: 300px; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); transform: translateY(-4px); }
.contract-container { padding: 1em; border: 1px solid #ddd; background-color: #fdfdfd; font-family: 'Courier New', monospace; font-size: 15px; line-height: 1.6; color: #333; white-space: pre-wrap; word-wrap: break-word; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
.contract-container::-webkit-scrollbar { width: 8px; }
.contract-container::-webkit-scrollbar-thumb { background-color: #ccc; border-radius: 4px; }
.progress-container-style { height: 300px; overflow-y: scroll; border: 1px solid #ccc; background-color: #f0f2f6; padding: 10px; font-family: monospace; font-size: 0.9rem; margin-bottom: 1rem; white-space: pre-wrap; color: #333; }
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# --- UI Layout ---
col1, col2 = st.columns([6, 4])
analyze_button_state = None

# --- State 1: Input (Largely Unchanged) ---
if not st.session_state.analysis_running and not st.session_state.final_report and not st.session_state.error_message:
    # logger.info("Rendering Input State.") # App logger, not workflow
    with col1:
        st.subheader("Nhập Hợp đồng")
        input_method = st.radio("Phương thức nhập:", ("Tải file lên", "Dán văn bản"), horizontal=True, key="input_method_radio")
        # ... (rest of your input logic is fine) ...
        contract_text_input = "" 
        uploaded_file = None
        if input_method == "Tải file lên":
            uploaded_file = st.file_uploader("Tải hợp đồng (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="file_uploader")
            if uploaded_file:
                if st.session_state.get('last_uploaded_file_id') != uploaded_file.id:
                    file_bytes = uploaded_file.getvalue()
                    file_name = uploaded_file.name
                    try:
                        with st.spinner(f"Đang trích xuất văn bản từ {file_name}..."):
                            contract_text_input = utils.extract_text(file_bytes, file_name)
                        logger.info(f"Trích xuất thành công từ {file_name}.")
                        st.session_state.original_contract_text = contract_text_input
                        st.session_state.last_uploaded_file_id = uploaded_file.id
                    except ValueError as ve: st.error(f"Lỗi file: {ve}"); logger.error(f"File extraction error: {ve}", exc_info=True); st.session_state.original_contract_text = ""
                    except Exception as e: st.error(f"Lỗi không xác định khi xử lý file: {e}"); logger.error(f"Unexpected file processing error: {e}", exc_info=True); st.session_state.original_contract_text = ""
                preview_text = st.session_state.original_contract_text[:1000] + ("..." if len(st.session_state.original_contract_text) > 1000 else "")
                st.text_area("Xem trước nội dung:", preview_text, height=150, disabled=True, key="text_preview_upload")
        elif input_method == "Dán văn bản":
            text_input_area_val = st.text_area("Dán nội dung hợp đồng vào đây:", value=st.session_state.original_contract_text, height=400, key="text_input_area")
            if text_input_area_val.strip() != st.session_state.original_contract_text.strip():
                 st.session_state.original_contract_text = text_input_area_val.strip()
        analyze_button_state = st.button("🔍 Phân tích Hợp đồng", key="analyze_button", disabled=(not st.session_state.original_contract_text.strip()), use_container_width=True, type="primary")
    with col2:
        st.markdown("### Hướng dẫn")
        st.markdown("""
            1.  **Tải file** hoặc **Dán văn bản**.
            2.  Nhấn **Phân tích Hợp đồng**.
            3.  **Log chi tiết** sẽ hiện ở cột này trong quá trình chạy.
            4.  Xem kết quả.
            """)
        st.info("Lưu ý: Kết quả phân tích chỉ mang tính tham khảo.")

# --- ACTION: Handle Button Click (Modified log placeholder handling) ---
if analyze_button_state:
    logger.info("Analyze button clicked.") # App logger
    if st.session_state.original_contract_text and st.session_state.original_contract_text.strip():
        st.session_state.analysis_running = True
        st.session_state.final_report = None
        st.session_state.error_message = None
        st.session_state.progress_updates = ["Bắt đầu phân tích..."]
        st.session_state[LOG_LIST_KEY] = [] # Clear log list for new run
        
        # Ensure the placeholder widget exists for the handler to use.
        # If it does, clear its current content.
        current_placeholder_widget = st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY)
        if current_placeholder_widget:
             # Use the handler's format_log_html method to set initial content
            retrieved_handler = st.session_state.get(LOG_HANDLER_INSTANCE_KEY)
            if retrieved_handler:
                initial_content = retrieved_handler.format_log_html(["Đang chờ log đầu tiên..."])
                current_placeholder_widget.markdown(initial_content, unsafe_allow_html=True)
            else: # Fallback if handler somehow not in session state (should not happen)
                current_placeholder_widget.markdown("<div class='progress-container-style'>Đang chờ log đầu tiên...</div>", unsafe_allow_html=True)

        st.session_state.annotations_data = []
        st.session_state.async_task_started_for_current_analysis = False
        logger.info("Set analysis_running=True. Triggering rerun for analysis phase.")
        st.rerun()
    else:
        st.warning("Vui lòng cung cấp nội dung hợp đồng trước khi phân tích.")
        logger.warning("Analyze clicked but no contract text found.")


# --- State 2: Analysis Running (Modified log placeholder handling) ---
elif st.session_state.analysis_running:
    # logger.info(f"Rendering 'Analysis Running' state. Async task started: {st.session_state.async_task_started_for_current_analysis}") # App logger
    
    with col1:
        st.info("⏳ Quá trình phân tích đang diễn ra...")
        st.write("Vui lòng chờ. Kết quả sẽ hiển thị sau khi hoàn tất.")

    with col2:
        st.subheader("Trạng thái Phân tích")
        progress_area_placeholder = st.empty() 
        with progress_area_placeholder.container():
            # ... (progress bar logic unchanged) ...
            current_step_count = len(st.session_state.progress_updates)
            total_steps_estimate = 5
            progress_value = min(1.0, (current_step_count) / total_steps_estimate)
            st.progress(progress_value)
            last_progress = st.session_state.progress_updates[-1] if st.session_state.progress_updates else "Khởi tạo..."
            st.text(f"Trạng thái: {last_progress}")
            st.spinner("Đang xử lý...")

        st.markdown("---")
        st.subheader("Workflow Log (Live)")
        
        # Ensure the st.empty() placeholder for logs is created and stored in session_state
        # if it doesn't exist yet for this "analysis_running" state.
        if st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY) is None:
            st.session_state[LOG_PLACEHOLDER_WIDGET_KEY] = st.empty()
            logger.info(f"Created/Assigned log placeholder widget in session_state['{LOG_PLACEHOLDER_WIDGET_KEY}']")
            # Display initial content
            retrieved_handler_for_initial_display = st.session_state.get(LOG_HANDLER_INSTANCE_KEY)
            if retrieved_handler_for_initial_display:
                initial_log_html = retrieved_handler_for_initial_display.format_log_html(
                    st.session_state.get(LOG_LIST_KEY, []) or ["Đang chờ log đầu tiên..."]
                )
                st.session_state[LOG_PLACEHOLDER_WIDGET_KEY].markdown(initial_log_html, unsafe_allow_html=True)
        # else:
            # The placeholder already exists, the handler will update it.
            # logger.debug(f"Log placeholder widget '{LOG_PLACEHOLDER_WIDGET_KEY}' already exists.")
            # We can optionally ensure its content reflects the current log list if a non-handler rerun happened.
            # retrieved_handler = st.session_state.get(LOG_HANDLER_INSTANCE_KEY)
            # if retrieved_handler and st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY):
            #     current_log_html = retrieved_handler.format_log_html(st.session_state.get(LOG_LIST_KEY, []))
            #     st.session_state[LOG_PLACEHOLDER_WIDGET_KEY].markdown(current_log_html, unsafe_allow_html=True)
            

    async def run_analysis_async_wrapper():
        # workflow_logger is used INSIDE this function by the actual workflow.
        # The handler is already attached to workflow_logger.
        workflow_logger.info("ASYNC_WRAPPER: Task starting...") # This should appear once.
        try:
            def streamlit_progress_callback(event: ProgressEvent):
                if event.progress not in st.session_state.progress_updates:
                    st.session_state.progress_updates.append(event.progress)

            workflow = MultiAgentContractReviewWorkflow(
                timeout=config.WORKFLOW_TIMEOUT,
                verbose=config.VERBOSE_WORKFLOW,
                progress_callback=streamlit_progress_callback
            )
            start_event = WorkflowStartEvent(
                contract_text=st.session_state.original_contract_text,
                query=config.DEFAULT_QUERY,
                tools=[law_tool] if law_tool else [],
                index=my_index
            )
            # Log with workflow_logger so StreamlitLogHandler picks it up
            workflow_logger.info(f"ASYNC_WRAPPER: Calling workflow.run for query: '{start_event.query[:50]}...'")
            
            result_output = await workflow.run(**start_event.model_dump())
            workflow_logger.info("ASYNC_WRAPPER: workflow.run has completed.")

            if isinstance(result_output, FinalOutputEvent):
                # ... (rest of result processing is fine) ...
                final_output : FinalOutputEvent = result_output
                st.session_state.final_report = final_output.report
                st.session_state.annotations_data = final_output.annotations
                st.session_state.error_message = None
                workflow_logger.info(f"ASYNC_WRAPPER: Successfully processed FinalOutputEvent. Report: {len(final_output.report if final_output.report else '')}, Annotations: {len(final_output.annotations if final_output.annotations else [])}")

            else:
                error_info = f"Workflow finished with unexpected result type: {type(result_output)}"
                workflow_logger.error(f"ASYNC_WRAPPER: {error_info}")
                st.session_state.error_message = error_info; st.session_state.final_report = None; st.session_state.annotations_data = []
        except Exception as e:
            error_msg = f"Lỗi thực thi Workflow: {e}"
            workflow_logger.error(f"ASYNC_WRAPPER: {error_msg}", exc_info=True) # Log with stack trace
            st.session_state.error_message = error_msg; st.session_state.final_report = None; st.session_state.annotations_data = []
        finally:
            workflow_logger.info("ASYNC_WRAPPER: Task finished. Setting analysis_running to False.")
            st.session_state.analysis_running = False
            st.session_state.async_task_started_for_current_analysis = False
            st.rerun()

    if not st.session_state.async_task_started_for_current_analysis:
        logger.info("Launch check: Async task launching...") # App logger
        st.session_state.async_task_started_for_current_analysis = True
        asyncio.run(run_analysis_async_wrapper())
        # logger.info("Launch check: asyncio.run has completed.") # This line is often not reached before rerun
    # else:
        # logger.info("Launch check: Async task already running.") # App logger


# --- State 3: Final View (Modified log placeholder cleanup and final log display) ---
elif not st.session_state.analysis_running and (st.session_state.final_report or st.session_state.error_message):
    # logger.info("Rendering Final State (Results or Error).") # App logger
    
    # Explicitly clear the live log placeholder from the UI
    log_placeholder_widget_final = st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY)
    if log_placeholder_widget_final:
        log_placeholder_widget_final.empty() # Remove from display
        st.session_state[LOG_PLACEHOLDER_WIDGET_KEY] = None # Clear from session
        logger.info(f"Cleared and removed live log placeholder '{LOG_PLACEHOLDER_WIDGET_KEY}'.")

    # ... (rest of your result/error display logic is fine) ...
    original_text = st.session_state.get('original_contract_text', "")
    report_text = st.session_state.get('final_report', "")
    error_message = st.session_state.get('error_message', "")
    annotations_data = st.session_state.get('annotations_data', [])

    if error_message:
        # ... error display logic ...
        with col1: st.error(f"**Lỗi Phân tích!**"); st.error(error_message)
        with col2: st.warning("Đã xảy ra lỗi. Xem chi tiết bên trái.")
    elif report_text:
        # ... report display logic ...
        try:
            with st.spinner("Đang xử lý highlight..."): highlighted_contract_html = utils.highlight_contract_with_annotations(original_text, annotations_data)
        except Exception as high_err: highlighted_contract_html = f'<div class="contract-container">{html.escape(original_text).replace(chr(10), "<br>")}</div>'; st.error(f"Lỗi highlight: {high_err}")
        with col1: st.subheader("Văn bản Hợp đồng (Highlight)"); st.markdown(f'<div class="contract-container">{highlighted_contract_html}</div>', unsafe_allow_html=True)
        with col2: st.subheader("Báo cáo Phân tích"); st.markdown(report_text, unsafe_allow_html=True)


    # Display final logs in an expander using the handler's formatting
    log_messages_final_list = st.session_state.get(LOG_LIST_KEY, [])
    if log_messages_final_list:
        retrieved_handler_for_final_logs = st.session_state.get(LOG_HANDLER_INSTANCE_KEY)
        if retrieved_handler_for_final_logs:
            st.markdown("---")
            with st.expander("Xem Log Phân tích Chi tiết (Hoàn chỉnh)", expanded=bool(error_message)):
                final_log_html_content = retrieved_handler_for_final_logs.format_log_html(log_messages_final_list)
                st.markdown(final_log_html_content, unsafe_allow_html=True)
        else: # Fallback if handler instance is missing
            st.markdown("---")
            with st.expander("Xem Log Phân tích Chi tiết (Raw)", expanded=bool(error_message)):
                st.code("".join(log_messages_final_list), language='log')


    if st.button("Phân tích Hợp đồng Khác", key="new_analysis_button_final", use_container_width=True):
        # ... (reset logic is fine, ensure placeholder is cleared) ...
        st.session_state.analysis_running = False; st.session_state.final_report = None; st.session_state.error_message = None
        st.session_state.progress_updates = []; st.session_state[LOG_LIST_KEY] = []
        st.session_state.original_contract_text = ""; st.session_state.annotations_data = []
        st.session_state.async_task_started_for_current_analysis = False
        if 'file_uploader' in st.session_state: del st.session_state['file_uploader']
        if 'last_uploaded_file_id' in st.session_state: del st.session_state['last_uploaded_file_id']
        
        # Ensure placeholder is cleared on new analysis
        placeholder_on_reset = st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY)
        if placeholder_on_reset:
            placeholder_on_reset.empty()
            st.session_state[LOG_PLACEHOLDER_WIDGET_KEY] = None
        logger.info("Resetting state for new analysis.")
        st.rerun()

# --- Fallback State (Ensure placeholder cleanup) ---
else:
    # logger.debug("Rendering fallback/initial state.") # App logger
    placeholder_fallback = st.session_state.get(LOG_PLACEHOLDER_WIDGET_KEY)
    if placeholder_fallback:
        placeholder_fallback.empty()
        st.session_state[LOG_PLACEHOLDER_WIDGET_KEY] = None
        logger.info(f"Cleared log placeholder '{LOG_PLACEHOLDER_WIDGET_KEY}' in fallback state.")
    pass
# workflow.py
import logging
import json
import asyncio
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable

# LlamaIndex Imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
)
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.llms import ChatMessage, MessageRole

# Qdrant Client
from qdrant_client import AsyncQdrantClient, QdrantClient

# Project Imports
from prompts import PromptStore
import config # Assuming config.py is in the same directory or accessible via PYTHONPATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Event Definitions ---

class WorkflowStartEvent(StartEvent):
    contract_text: str
    query: str
    tools: List[QueryEngineTool]

class ContractTextEvent(Event):
    contract_text: str

class ContextAnalyzedEvent(Event):
    contract_type: Optional[str]
    key_elements: List[str]
    cited_legal_documents: List[str]
    contract_text: str

class AnalysisResultsEvent(Event):
    legal_issues: str
    logic_issues: str
    risk_issues: str

class FinalOutputEvent(Event):
    report: str

class ProgressEvent(Event):
    progress: str

# --- Workflow Definition ---

class MultiAgentContractReviewWorkflow(Workflow):

    def __init__(
        self,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        **kwargs,
    ):
        """
        Initialize the workflow.

        Args:
            progress_callback: An optional function to call with ProgressEvent updates.
            **kwargs: Additional arguments for the base Workflow class.
        """
        super().__init__(**kwargs)
        self.progress_callback = progress_callback
        self.prompt_store = PromptStore()   

    def _send_progress(self, ctx: Context, message: str):
        """Helper to send progress updates via callback and stream."""
        event = ProgressEvent(progress=message)
        ctx.write_event_to_stream(event)
        if self.progress_callback:
            try:
                # Ensure callback runs in the main thread if needed by UI framework
                # This might require asyncio.get_event_loop().call_soon_threadsafe or similar
                # For simplicity, calling directly here. Test with Streamlit.
                self.progress_callback(event)
            except Exception as e:
                 logger.error(f"Error in progress callback: {e}")


    @step()
    async def preprocess_input(self, ctx: Context, ev: WorkflowStartEvent) -> ContractTextEvent:
        """Stores initial context data."""
        logger.info("Starting step: preprocess_input")
        await ctx.set("query", ev.query)
        await ctx.set("tools", ev.tools)
        await ctx.set("contract_text", ev.contract_text)

        self._send_progress(ctx, "Preprocessing completed.")
        return ContractTextEvent(contract_text=ev.contract_text)

    @step()
    async def analyze_context(self, ctx: Context, ev: ContractTextEvent) -> ContextAnalyzedEvent:
        """Analyzes the contract text to extract high-level context."""
        logger.info("Starting step: analyze_context")
        query = await ctx.get("query")
        contract_text = ev.contract_text

        # --- Prompt for Context Analysis (Same as notebook) ---
        # --- Get Prompt from Store ---
        prompt = self.prompt_store.get_context_analysis_prompt(
            query=query,
            contract_text=contract_text
        )
        # --- End Prompt ---

        try:
            response = await Settings.llm.acomplete(prompt)
            logger.debug(f"Context analysis LLM raw response: {response}")
            parsed = json.loads(str(response)) # Ensure response is stringified before parsing

            contract_type = parsed.get("contract_type")
            key_elements = parsed.get("key_elements", [])
            cited_laws = parsed.get("cited_legal_documents", [])

            # Ensure lists contain strings
            key_elements = [str(item) for item in key_elements if item is not None]
            cited_laws = [str(item) for item in cited_laws if item is not None]


        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from context analysis LLM response: {e}\nResponse was: {response}")
            # Fallback or raise error
            contract_type = "Error parsing response"
            key_elements = []
            cited_laws = []
        except Exception as e:
            logger.error(f"Error during context analysis: {e}")
            # Fallback or raise error
            contract_type = "Error during analysis"
            key_elements = []
            cited_laws = []


        await ctx.set("contract_type", contract_type)
        await ctx.set("key_elements", key_elements)
        await ctx.set("cited_laws", cited_laws)

        self._send_progress(ctx, "Context analysis completed.")
        return ContextAnalyzedEvent(
            contract_type=contract_type,
            key_elements=key_elements,
            cited_legal_documents=cited_laws,
            contract_text=contract_text,
        )

    @step()
    async def detailed_analysis(self, ctx: Context, ev: ContextAnalyzedEvent) -> AnalysisResultsEvent:
        """Performs detailed logic, risk, and legal (RAG-enhanced) analysis."""
        logger.info("Starting step: detailed_analysis")
        contract_text = ev.contract_text
        query = await ctx.get("query")
        contract_type = ev.contract_type
        cited_laws_list = ev.cited_legal_documents

        # --- Logic, Risk, Legal Prompts (Same as notebook) ---
        # --- Get Prompts from Store ---
        logic_prompt = self.prompt_store.get_logic_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        risk_prompt = self.prompt_store.get_risk_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        legal_prompt = self.prompt_store.get_legal_review_prompt(
            contract_text=contract_text, cited_laws_list=cited_laws_list
        )
        # --- End Prompts ---


        self._send_progress(ctx, "Running detailed analysis (LLM)...")
        try:
            logic_response, risk_response, legal_review_response = await asyncio.gather(
                Settings.llm.acomplete(logic_prompt),
                Settings.llm.acomplete(risk_prompt),
                Settings.llm.acomplete(legal_prompt)
            )

            logic_issues_text = logic_response.text
            risk_issues_text = risk_response.text
            legal_review_text = legal_review_response.text

            logger.debug(f"Logic analysis response: {logic_issues_text}")
            logger.debug(f"Risk analysis response: {risk_issues_text}")
            logger.debug(f"Initial legal review response: {legal_review_text}")

        except Exception as e:
            logger.error(f"Error calling LLM for detailed analysis: {e}")
            logic_issues_text = '{"logic_findings": [{"type": "error", "description": "Error during analysis", "clauses_involved": []}]}'
            risk_issues_text = '{"risk_findings": [{"risk_type": "error", "description": "Error during analysis", "suggestion": "N/A"}]}'
            legal_review_text = "Error during initial legal review."


        self._send_progress(ctx, "Initial legal review completed. Starting RAG checks...")

        # --- Logic xử lý kết quả phân tích pháp lý và RAG (Same as notebook, with logging/error handling) ---
        final_legal_issues_output = ""
        legal_issues_results = []
        no_issues_found_msg = "Không xác định được điểm cụ thể nào trong văn bản cần đánh giá pháp lý thêm."

        if legal_review_text.strip() == no_issues_found_msg:
            final_legal_issues_output = no_issues_found_msg
            logger.info("LLM determined no specific legal points require further checking.")
        elif "Error during initial legal review." in legal_review_text :
            final_legal_issues_output = legal_review_text # Propagate error
            logger.error("Skipping RAG due to error in initial legal review.")
        else:
            try:
                agent_tools = await ctx.get("tools")
                if not agent_tools:
                     raise ValueError("RAG Tool not found in context.")

                agent = FunctionCallingAgent.from_tools(
                    agent_tools,
                    verbose=config.VERBOSE_WORKFLOW,
                    # llm=Settings.llm # Ensure agent uses the same LLM if needed
                )

                # --- Parse LLM's Legal Review Output (Robust Parsing Needed) ---
                potential_issues = []
                try:
                    # Improved parsing logic (example - adjust based on actual LLM output format consistency)
                    lines = [line.strip() for line in legal_review_text.split('\n') if line.strip()]
                    current_issue = {}
                    for line in lines:
                        if line.startswith("- **Điều khoản/Nội dung HĐ:**"):
                            if current_issue.get("clause_content") and current_issue.get("reason_for_check"):
                                potential_issues.append(current_issue) # Add completed previous issue
                                current_issue = {} # Reset
                            content = line.replace("- **Điều khoản/Nội dung HĐ:**", "").strip()
                            current_issue["clause_content"] = content
                        elif line.startswith("- **Lý do cần kiểm tra:**") and current_issue:
                            reason = line.replace("- **Lý do cần kiểm tra:**", "").strip()
                            current_issue["reason_for_check"] = reason
                            # Add issue once reason is found (assuming pair structure)
                            if current_issue.get("clause_content"):
                                potential_issues.append(current_issue)
                                current_issue = {} # Reset

                    # Handle dangling clause if file ends with it
                    if current_issue.get("clause_content") and not current_issue.get("reason_for_check"):
                         current_issue["reason_for_check"] = "Lý do không được LLM nêu rõ."
                         potential_issues.append(current_issue)
                    elif current_issue.get("clause_content") and current_issue.get("reason_for_check"):
                         # This case should be covered above, but as a safeguard
                         potential_issues.append(current_issue)

                    logger.info(f"Parsed {len(potential_issues)} potential legal issues for RAG check.")
                    if not potential_issues and legal_review_text.strip():
                        logger.warning("Legal review text was present but no issues parsed. Check LLM output format and parsing logic.")
                        # Maybe add the raw text as a single issue?
                        final_legal_issues_output = "Could not parse specific legal issues from LLM output. Raw LLM response:\n" + legal_review_text

                except Exception as parse_err:
                    logger.error(f"Error parsing legal review text: {parse_err}\nRaw Text: {legal_review_text}")
                    final_legal_issues_output = f"Error parsing legal review text: {parse_err}"
                    potential_issues = [] # Prevent further processing

                # --- Iterate and Query RAG Agent ---
                for i, issue in enumerate(potential_issues):
                    clause_content = issue.get("clause_content", "N/A")
                    reason = issue.get("reason_for_check", "N/A")
                    self._send_progress(ctx, f"Checking legal issue {i+1}/{len(potential_issues)}: '{clause_content[:50]}...'")

                    # --- Build RAG Query (Same smart logic as notebook) ---
                    query_for_rag = ""
                    # Simple heuristic to find potential citations (needs improvement)
                    potential_citation = clause_content # Default
                    if "Điều" in clause_content and ("Luật" in clause_content or "Nghị định" in clause_content):
                         # Try to extract like "Điều X [khoản Y] [điểm Z] Luật/Nghị định ABC"
                         # This is complex NLP, keeping it simple for now
                         potential_citation = clause_content # Use the whole clause text for now

                    if "không chính xác" in reason or "không tồn tại" in reason:
                        query_for_rag = f"Xác minh tính chính xác của trích dẫn pháp luật sau đây được nêu trong hợp đồng: '{potential_citation}'. Trích dẫn này có tồn tại không và nội dung chính là gì theo luật Việt Nam liên quan đến lao động?"
                    elif "mâu thuẫn" in reason or "không nhất quán" in reason or "diễn giải sai" in reason:
                        query_for_rag = f"Nội dung hợp đồng sau: '{clause_content}' được cho là '{reason}'. Cung cấp thông tin pháp luật lao động Việt Nam liên quan để đối chiếu sự mâu thuẫn/không nhất quán/diễn giải sai này."
                    elif "thiếu chi tiết" in reason or "thiếu thông tin" in reason:
                        query_for_rag = f"Nội dung hợp đồng sau: '{clause_content}' được cho là '{reason}'. Luật lao động Việt Nam yêu cầu những chi tiết gì thêm cho vấn đề này?"
                    else: # General check
                        query_for_rag = f"Kiểm tra pháp lý cho nội dung hợp đồng sau: '{clause_content}'. Lý do cần kiểm tra: '{reason}'. Cung cấp thông tin từ luật lao động Việt Nam liên quan để đối chiếu."

                    logger.info(f"Querying RAG for issue: {clause_content} | Reason: {reason}")
                    logger.debug(f"RAG Query: {query_for_rag}")

                    try:
                        response = await agent.aquery(query_for_rag)
                        logger.debug(f"RAG Agent raw response: {response}")

                        # --- Process RAG Response (Same logic as notebook) ---
                        relevant_response = response.response if hasattr(response, 'response') else str(response) # Handle different response types
                        citation_text = "Không có nguồn tham khảo cụ thể được tìm thấy."
                        rag_status = "unknown"

                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            citations = defaultdict(set)
                            valid_citation_found = False
                            # Prepare filenames from cited laws list for matching
                            cited_laws_files = {f + ".pdf" for f in cited_laws_list if isinstance(f, str)} # Assuming PDF format
                            cited_laws_files.update({f.replace(" ", "_") + ".pdf" for f in cited_laws_list if isinstance(f, str)}) # Add underscore variant

                            logger.debug(f"Looking for citations matching: {cited_laws_files}")

                            for node in response.source_nodes:
                                metadata = node.node.metadata if node.node else {}
                                page_number = metadata.get("page_label", "N/A")
                                file_name = metadata.get("file_name", "Unknown file")

                                logger.debug(f"RAG Source Node: File='{file_name}', Page='{page_number}'")

                                # Prioritize files mentioned in the contract's cited list
                                if file_name in cited_laws_files:
                                    citations[file_name].add(str(page_number)) # Ensure page is string
                                    valid_citation_found = True
                                    logger.debug(f"Matched cited law: {file_name}")
                                # Optional: Include other relevant sources even if not explicitly cited?
                                # else:
                                #    citations[file_name].add(str(page_number))

                            if citations: # If any citations were found (even if not matching cited list initially)
                                citation_text = "Nguồn tham khảo: " + "; ".join(
                                    f"{file} (trang {', '.join(sorted(list(pages)))})"
                                    for file, pages in citations.items() if pages
                                )
                                if not valid_citation_found:
                                     citation_text += " (Lưu ý: Nguồn này không được liệt kê trong các văn bản trích dẫn ban đầu của hợp đồng)"
                                     logger.warning(f"RAG found sources, but not matching the initial cited list for issue: {clause_content}")

                            # --- Evaluate RAG response content (Same logic as notebook) ---
                            response_lower = relevant_response.lower() if relevant_response else ""
                            if "không tìm thấy thông tin" in response_lower or \
                               "không có điều khoản" in response_lower or \
                               "không có quy định" in response_lower or \
                               "không tồn tại" in response_lower or \
                               "không xác định" in response_lower or \
                               "không thể cung cấp" in response_lower:
                                rag_status = "citation_not_found"
                            elif reason and ("mâu thuẫn" in reason or "không nhất quán" in reason or "diễn giải sai" in reason):
                                # Check if RAG response confirms or denies the contradiction
                                # This requires more sophisticated analysis of the RAG text itself
                                rag_status = "potential_contradiction" # Keep as potential for now
                            else:
                                rag_status = "found_info"
                        else: # No source nodes
                            response_lower = relevant_response.lower() if relevant_response else ""
                            if "không tìm thấy" in response_lower or "không có thông tin" in response_lower:
                                rag_status = "citation_not_found"
                            logger.warning(f"RAG Agent returned no source nodes for query: {query_for_rag}")


                        # --- Format Legal Issue Output ---
                        legal_issues_results.append(f"**Điểm cần lưu ý:** {clause_content}")
                        legal_issues_results.append(f"  * **Lý do kiểm tra (từ LLM):** {reason}")

                        if rag_status == "citation_not_found":
                            legal_issues_results.append(f"  * **Kết quả tra cứu (từ RAG):** {relevant_response}. **=> Cần kiểm tra lại tính chính xác của nội dung/trích dẫn này trong hợp đồng.**")
                        elif rag_status == "potential_contradiction":
                            legal_issues_results.append(f"  * **Thông tin tra cứu (từ RAG):** {relevant_response}")
                            legal_issues_results.append(f"  * **Đánh giá:** Thông tin tra cứu được có thể khác biệt/làm rõ vấn đề '{reason}'. **=> Cần đối chiếu kỹ lưỡng giữa hợp đồng và luật.**")
                        elif rag_status == "found_info":
                            legal_issues_results.append(f"  * **Thông tin tra cứu (từ RAG):** {relevant_response}")
                            legal_issues_results.append(f"  * **Đánh giá:** Đã tìm thấy thông tin liên quan. Cần người dùng xem xét sự phù hợp và tuân thủ.")
                        else: # unknown or other cases
                            legal_issues_results.append(f"  * **Thông tin tra cứu (từ RAG):** {relevant_response}")

                        legal_issues_results.append(f"  * **{citation_text}**")
                        legal_issues_results.append("---")

                    except Exception as rag_err:
                        logger.error(f"Error querying RAG agent for issue '{clause_content}': {rag_err}")
                        legal_issues_results.append(f"**Điểm cần lưu ý:** {clause_content}")
                        legal_issues_results.append(f"  * **Lý do kiểm tra (từ LLM):** {reason}")
                        legal_issues_results.append(f"  * **Lỗi tra cứu:** Không thể hoàn thành tra cứu tự động cho điểm này do lỗi: {rag_err}")
                        legal_issues_results.append("---")

                # --- Finalize Legal Issues Output ---
                if not legal_issues_results and not final_legal_issues_output: # Check if it wasn't already set by parsing error
                     # This case means parsing found issues, but RAG failed for all or formatted output is empty
                     final_legal_issues_output = "Đã xác định các điểm cần kiểm tra pháp lý, nhưng quá trình tra cứu hoặc định dạng kết quả không thành công."
                     logger.warning("Potential issues were identified, but RAG/formatting resulted in empty output.")
                elif legal_issues_results :
                     final_legal_issues_output = "\n".join(legal_issues_results)

            except Exception as agent_err:
                 logger.error(f"Error initializing or using RAG agent: {agent_err}")
                 final_legal_issues_output = f"Lỗi nghiêm trọng trong quá trình kiểm tra pháp lý tự động: {agent_err}"

        self._send_progress(ctx, "Detailed analysis completed.")
        logger.info("Finished step: detailed_analysis")
        return AnalysisResultsEvent(
            legal_issues=final_legal_issues_output,
            logic_issues=logic_issues_text,
            risk_issues=risk_issues_text
        )

    @step()
    async def synthesize_output(self, ctx: Context, ev: AnalysisResultsEvent) -> StopEvent:
        """Synthesizes the final report from all analysis steps."""
        logger.info("Starting step: synthesize_output")
        self._send_progress(ctx, "Synthesizing final report...")

        # Retrieve context items (Ensure they exist)
        contract_type = await ctx.get("contract_type", "Không xác định")
        key_elements = await ctx.get("key_elements", [])
        cited_laws = await ctx.get("cited_laws", [])
        contract_text = await ctx.get("contract_text", "N/A")
        query = await ctx.get("query", "N/A")

        context_info = {
            "contract_type": contract_type,
            "key_elements": key_elements,
            "cited_laws": cited_laws
        }
        no_issues_found_msg = "Không xác định được điểm cụ thể nào trong văn bản cần đánh giá pháp lý thêm."

        # --- Synthesis Prompt (Same as notebook, uses new variables) ---
        # --- Get Prompt from Store ---
        prompt = self.prompt_store.get_synthesis_prompt(
            context_info=context_info,
            query=query,
            legal_issues=ev.legal_issues,
            logic_issues=ev.logic_issues,
            risk_issues=ev.risk_issues,
            contract_text=contract_text # Pass original contract text
        )
        # --- End Prompt ---

        try:
            output = await Settings.llm.acomplete(prompt)
            report = output.text
            logger.debug(f"Synthesis LLM raw response: {report}")
        except Exception as e:
            logger.error(f"Error calling LLM for synthesis: {e}")
            report = f"# Lỗi Tổng hợp Báo cáo\n\nKhông thể tạo báo cáo do lỗi: {e}\n\n## Dữ liệu phân tích:\n### Pháp lý:\n{ev.legal_issues}\n### Logic:\n{ev.logic_issues}\n### Rủi ro:\n{ev.risk_issues}"

        self._send_progress(ctx, "Synthesis completed.")
        logger.info("Finished step: synthesize_output")
        # Return the final report within a StopEvent containing FinalOutputEvent
        return StopEvent(result=FinalOutputEvent(report=report.strip()))

# --- Initialization Function ---

async def initialize_analysis_system():
    """Initializes LLM, Embeddings, Vector Store, Index, and Tool. Returns the RAG tool."""
    logger.info("Initializing LlamaIndex Settings and Qdrant connection...")
    try:
        # Configure LlamaIndex Settings
        Settings.llm = Ollama(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=config.REQUEST_TIMEOUT
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=config.EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=config.REQUEST_TIMEOUT
        )
        logger.info(f"LLM ({config.LLM_MODEL}) and Embeddings ({config.EMBED_MODEL}) configured.")

        # Initialize Qdrant Client (Async for workflow, Sync might be needed for store init depending on version)
        logger.info(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}...")
        async_client = AsyncQdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        # Optionally test connection or get collection info
        # collection_info = await async_client.get_collection(config.QDRANT_COLLECTION)
        # logger.info(f"Qdrant collection '{config.QDRANT_COLLECTION}' info: {collection_info}")

        vector_store = QdrantVectorStore(
            aclient=async_client, # Use async client
            collection_name=config.QDRANT_COLLECTION
        )
        logger.info(f"QdrantVectorStore initialized for collection '{config.QDRANT_COLLECTION}'.")

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("VectorStoreIndex created from Qdrant store.")

        query_engine = index.as_query_engine(similarity_top_k=config.SIMILARITY_TOP_K)
        logger.info(f"Query engine created with similarity_top_k={config.SIMILARITY_TOP_K}.")

        law_tool = QueryEngineTool.from_defaults(
            query_engine,
            name="labor_law_tool",
            description="Công cụ truy vấn thông tin chi tiết về các quy định pháp luật lao động tại Việt Nam, bao gồm Bộ luật Lao động, Luật Việc làm, Luật An toàn vệ sinh lao động, và các văn bản liên quan."
        )
        logger.info("RAG QueryEngineTool 'labor_law_tool' created.")
        logger.info("Initialization complete.")
        return law_tool

    except Exception as e:
        logger.error(f"Failed to initialize analysis system: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller
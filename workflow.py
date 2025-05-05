# workflow.py
import logging
import json
import asyncio
import re # Import regex
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable

# LlamaIndex Imports
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.tools import QueryEngineTool, BaseTool
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
)
# Qdrant Client
from qdrant_client import AsyncQdrantClient, QdrantClient

# Project Imports
from prompts import PromptStore # <<< Giả sử bạn đã cập nhật prompts.py
from utils import extract_json_from_response # Import utility
import config # Import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
workflow_step_logger = logging.getLogger("WorkflowSteps")
workflow_step_logger.setLevel(logging.INFO)


# --- Event Definitions (Giữ nguyên từ lần tối ưu trước) ---

class WorkflowStartEvent(StartEvent):
    contract_text: str
    query: str
    tools: List[BaseTool]

class ContractTextEvent(Event):
    contract_text: str

class ContextAnalyzedEvent(Event):
    contract_type: Optional[str]
    key_elements: List[str]
    cited_legal_documents: List[str]
    contract_text: str

class AnalysisResultsEvent(Event):
    verified_legal_issues: List[Dict[str, Any]]
    logic_issues: str # Raw JSON string from LLM (Cần parse để lấy exact_quote)
    risk_issues: str  # Raw JSON string from LLM (Cần parse để lấy exact_quote)

class FinalOutputEvent(Event):
    report: str
    annotations: List[Dict[str, Any]]

class ProgressEvent(Event):
    progress: str

# --- Workflow Definition ---

class MultiAgentContractReviewWorkflow(Workflow):

    def __init__(
        self,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.progress_callback = progress_callback
        self.prompt_store = PromptStore() # Initialize PromptStore with updated prompts

    def _send_progress(self, ctx: Context, message: str):
        # (Giữ nguyên logic _send_progress)
        event = ProgressEvent(progress=message)
        if hasattr(ctx, 'write_event_to_stream'): ctx.write_event_to_stream(event)
        if self.progress_callback:
            try:
                if callable(self.progress_callback):
                    loop = asyncio.get_running_loop()
                    if loop.is_running(): loop.call_soon_threadsafe(self.progress_callback, event)
                    else: self.progress_callback(event)
                else: logger.warning(f"Progress callback is not callable: {type(self.progress_callback)}")
            except Exception as e: logger.error(f"Error executing progress callback: {e}", exc_info=True)

    @step()
    async def preprocess_input(self, ctx: Context, ev: WorkflowStartEvent) -> ContractTextEvent:
        # (Giữ nguyên)
        workflow_step_logger.info("Starting step: preprocess_input")
        await ctx.set("query", ev.query)
        await ctx.set("tools", ev.tools)
        await ctx.set("contract_text", ev.contract_text)
        self._send_progress(ctx, "Preprocessing completed.")
        return ContractTextEvent(contract_text=ev.contract_text)

    @step()
    async def analyze_context(self, ctx: Context, ev: ContractTextEvent) -> ContextAnalyzedEvent:
        # (Giữ nguyên logic, dùng prompt từ store)
        workflow_step_logger.info("Starting step: analyze_context")
        query = await ctx.get("query")
        contract_text = ev.contract_text
        prompt = self.prompt_store.get_context_analysis_prompt(query=query, contract_text=contract_text)
        try:
            response = await Settings.llm.acomplete(prompt)
            response_text = response.text
            workflow_step_logger.info("Context analysis LLM call completed.")
            logger.debug(f"Context analysis LLM raw response:\n{response_text}")
            parsed = extract_json_from_response(response_text)
            if not parsed: raise ValueError("Failed context analysis JSON extraction.")
            contract_type = parsed.get("contract_type")
            key_elements_raw = parsed.get("key_elements", [])
            cited_laws_raw = parsed.get("cited_legal_documents", [])
            key_elements = [str(item) for item in key_elements_raw if item]
            cited_laws = [str(item) for item in cited_laws_raw if item]
            workflow_step_logger.info(f"Context extracted: Type='{contract_type}', Elements={len(key_elements)}, Laws={len(cited_laws)}")
        except Exception as e:
            logger.error(f"Error during context analysis: {e}", exc_info=True)
            contract_type = "Lỗi Phân tích Ngữ cảnh"
            key_elements = []
            cited_laws = []
        await ctx.set("contract_type", contract_type)
        await ctx.set("key_elements", key_elements)
        await ctx.set("cited_laws", cited_laws)
        self._send_progress(ctx, "Context analysis completed.")
        return ContextAnalyzedEvent(
            contract_type=contract_type, key_elements=key_elements,
            cited_legal_documents=cited_laws, contract_text=contract_text,
        )

    @step()
    async def detailed_analysis(self, ctx: Context, ev: ContextAnalyzedEvent) -> AnalysisResultsEvent:
        """Performs detailed logic, risk, and legal analysis, aiming for verbatim quotes."""
        workflow_step_logger.info("Starting step: detailed_analysis")
        contract_text = ev.contract_text
        query = await ctx.get("query")
        contract_type = ev.contract_type
        cited_laws_list = ev.cited_legal_documents
        tools = await ctx.get("tools")

        # --- Get Updated Prompts from Store ---
        # <<< Đảm bảo PromptStore chứa các prompt đã cập nhật yêu cầu "exact_quote" / "Vị trí/Trích dẫn Nguyên văn" >>>
        logic_prompt = self.prompt_store.get_logic_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        risk_prompt = self.prompt_store.get_risk_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        legal_prompt = self.prompt_store.get_legal_review_prompt(
            contract_text=contract_text, cited_laws_list=cited_laws_list
        )

        # --- Call LLMs Concurrently ---
        self._send_progress(ctx, "Running detailed analysis (Logic, Risk, Legal Red Flags)...")
        try:
            logic_response, risk_response, legal_review_response = await asyncio.gather(
                Settings.llm.acomplete(logic_prompt),
                Settings.llm.acomplete(risk_prompt),
                Settings.llm.acomplete(legal_prompt),
                return_exceptions=True
            )

            # Handle potential errors from LLM calls
            default_logic_error = '{"logic_findings": [{"type": "error", "description": "Lỗi gọi LLM phân tích Logic", "exact_quote": []}]}'
            default_risk_error = '{"risk_findings": [{"risk_type": "error", "description": "Lỗi gọi LLM phân tích Rủi ro", "exact_quote": "", "suggestion": "N/A"}]}'
            default_legal_error = "Lỗi gọi LLM xác định điểm pháp lý ban đầu."

            if isinstance(logic_response, Exception):
                logger.error(f"Error in Logic Analysis LLM call: {logic_response}", exc_info=logic_response)
                logic_issues_text = default_logic_error
            else: logic_issues_text = logic_response.text

            if isinstance(risk_response, Exception):
                logger.error(f"Error in Risk Analysis LLM call: {risk_response}", exc_info=risk_response)
                risk_issues_text = default_risk_error
            else: risk_issues_text = risk_response.text

            if isinstance(legal_review_response, Exception):
                logger.error(f"Error in Legal Red Flag LLM call: {legal_review_response}", exc_info=legal_review_response)
                legal_review_text = default_legal_error
            else: legal_review_text = legal_review_response.text.strip()

            # print("=======================================")
            # print(legal_review_text)
            # print("=======================================")
            
            workflow_step_logger.info("Detailed analysis LLM calls completed.")
            logger.debug(f"Logic analysis response: {logic_issues_text[:200]}...")
            logger.debug(f"Risk analysis response: {risk_issues_text[:200]}...")
            logger.debug(f"Initial legal review response:\n{legal_review_text}")

        except Exception as e:
            logger.error(f"Critical error during concurrent LLM calls: {e}", exc_info=True)
            logic_issues_text = default_logic_error
            risk_issues_text = default_risk_error
            legal_review_text = "Lỗi hệ thống khi gọi LLM."

        # --- Process Legal Review and RAG ---
        self._send_progress(ctx, "Initial legal review processing...")
        verified_legal_issues_list: List[Dict[str, Any]] = []
        no_issues_found_msg = self.prompt_store.NO_LEGAL_ISSUES_FOUND_MSG

        if legal_review_text == no_issues_found_msg:
            workflow_step_logger.info("LLM determined no specific legal points require RAG check.")
        elif "Lỗi gọi LLM" in legal_review_text or "Lỗi hệ thống" in legal_review_text:
            logger.error(f"Skipping RAG due to error in initial legal review: {legal_review_text}")
            verified_legal_issues_list.append({
                "verbatim_quote": "Lỗi Phân tích Pháp lý Ban đầu", "initial_reason": legal_review_text,
                "rag_query": "N/A", "rag_result": "N/A", "verification_status": "ERROR",
                "verification_summary": "Không thể thực hiện tra cứu RAG.", "source_nodes": []
            })
        else:
            # Find the RAG tool
            rag_tool: Optional[BaseTool] = None
            for tool in tools:
                 if hasattr(tool, 'metadata') and tool.metadata.name == config.RAG_TOOL_NAME:
                      rag_tool = tool
                      break

            if not rag_tool:
                 logger.error(f"RAG tool '{config.RAG_TOOL_NAME}' not found.")
                 verified_legal_issues_list.append({
                     "verbatim_quote": "Lỗi Hệ thống", "initial_reason": "Công cụ RAG lỗi.",
                     "rag_query": "N/A", "rag_result": "N/A", "verification_status": "ERROR",
                     "verification_summary": "Lỗi cấu hình RAG.", "source_nodes": []
                 })
            else:
                 workflow_step_logger.info(f"Using RAG tool: {rag_tool.metadata.name}")
                 # Parse potential issues using the updated Markdown format
                 potential_issues = []
                 try:
                     # <<< CHANGE: Regex to capture 'Vị trí/Trích dẫn Nguyên văn' >>>
                     pattern = re.compile(
                        r"-\s*\*\*Vị trí/Trích dẫn Nguyên văn:\*\*\s*(.*?)\s*-?\s*\*\*Lý do cần kiểm tra:\*\*\s*(.*?)(?=\n-\s*\*\*Vị trí/Trích dẫn Nguyên văn:\*\*|\Z)",
                        re.DOTALL | re.IGNORECASE
                     )
                     matches = pattern.findall(legal_review_text)
                     workflow_step_logger.info(f"Regex matches found for legal issues: {len(matches)}")

                     for match in matches:
                         quote = match[0].strip() # This is the verbatim quote
                         reason = match[1].strip().replace(no_issues_found_msg, '').strip()
                         if quote and reason:
                             potential_issues.append({
                                 "verbatim_quote": quote, # Store the verbatim quote
                                 "reason_for_check": reason
                             })

                     if not matches and legal_review_text != no_issues_found_msg:
                         logger.warning("Could not parse legal red flags using updated regex. Check LLM output format.")
                         potential_issues.append({ "verbatim_quote": "Lỗi phân tích: Không tìm thấy định dạng Markdown mong muốn.", "reason_for_check": legal_review_text })

                 except Exception as parse_err:
                      logger.error(f"Error parsing legal review text: {parse_err}", exc_info=True)
                      verified_legal_issues_list.append({
                          "verbatim_quote": "Lỗi Phân tích Cú pháp", "initial_reason": f"Lỗi parse: {parse_err}",
                          "rag_query": "N/A", "rag_result": legal_review_text, "verification_status": "ERROR",
                          "verification_summary": "Không thể phân tích các điểm pháp lý từ LLM.", "source_nodes": []
                      })
                      potential_issues = [] # Stop processing

                 workflow_step_logger.info(f"Potential legal issues identified for RAG check: {len(potential_issues)}")
                 self._send_progress(ctx, f"Found {len(potential_issues)} potential legal issues. Starting RAG checks...")

                 # Iterate and Query RAG Tool Directly
                 for i, issue in enumerate(potential_issues):
                    # <<< CHANGE: Use verbatim_quote for context in RAG query if needed >>>
                    verbatim_quote = issue.get("verbatim_quote", "N/A")
                    reason = issue.get("reason_for_check", "N/A")
                    self._send_progress(ctx, f"Checking legal issue {i+1}/{len(potential_issues)}: '{verbatim_quote[:40]}...'")

                    rag_query = ""
                    rag_result_text = "Lỗi tra cứu RAG."
                    source_nodes_info = []
                    rag_response_successful = False

                    try:
                        # Build RAG Query (using verbatim_quote for context)
                        verification_focus = ""
                        reason_lower = reason.lower()
                        # (Logic to determine verification_focus remains the same)
                        if "không chính xác" in reason_lower or "không tồn tại" in reason_lower:
                            verification_focus = "Tập trung xác minh sự tồn tại và nội dung chính xác của trích dẫn này."
                        elif "mâu thuẫn" in reason_lower or "không nhất quán" in reason_lower or "diễn giải sai" in reason_lower or "sai luật" in reason_lower:
                             verification_focus = "Tập trung cung cấp nội dung luật liên quan để đối chiếu trực tiếp, làm rõ khả năng mâu thuẫn/sai khác."
                        elif "thiếu chi tiết" in reason_lower or "thiếu thông tin" in reason_lower:
                            law_match = re.search(r"(Luật|Bộ luật|Nghị định)[\s\w\/\-]+", reason, re.IGNORECASE)
                            law_ref = law_match.group(0).strip() if law_match else "luật liên quan"
                            verification_focus = f"Tập trung kiểm tra xem {law_ref} có yêu cầu bắt buộc các chi tiết đang bị thiếu không."
                        else:
                            verification_focus = "Cung cấp thông tin pháp lý liên quan nhất."

                        # Include verbatim quote in the query for better context
                        rag_query = f"Kiểm tra pháp lý cho phần hợp đồng sau: '{verbatim_quote}'. Lý do LLM gợi ý: '{reason}'. {verification_focus}"
                        workflow_step_logger.info(f"RAG Query {i+1}: {rag_query}")

                        # Call RAG tool directly
                        rag_response_obj = await rag_tool.acall({"input": rag_query})
                        # rag_response_obj = rag_response_obj.content
                        rag_response_successful = True
                        
                        if rag_response_obj.raw_output:
                            rag_result_text = rag_response_obj.content
                            rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                            print(f"[RAG Direct Response]: {rag_result_text}")
                            rag_response = rag_response_obj.raw_output
                            for node in rag_response.source_nodes:
                                metadata = node.node.metadata
                                source_nodes_info.append({
                                    "file_name": metadata.get("file_name", "N/A"),
                                    "page": metadata.get("page_label", "N/A"),
                                    "score": node.score # Lấy score nếu có
                                })
                            print(f"[RAG Source Nodes]: {source_nodes_info}")

                        # # Process response (same logic as previous optimized version)
                        # if isinstance(rag_response_obj, str):
                        #     rag_result_text = rag_response_obj
                        #     rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                        #     source_nodes_info = []
                        # elif hasattr(rag_response_obj, 'response'):
                        #     rag_result_text = rag_response_obj.response
                        #     rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                        #     if hasattr(rag_response_obj, 'source_nodes') and rag_response_obj.source_nodes:
                        #         for node in rag_response_obj.source_nodes:
                        #              metadata = node.node.metadata if node.node else {}
                        #              source_nodes_info.append({
                        #                  "file_name": metadata.get("file_name", "N/A"),
                        #                  "page": metadata.get("page_label", "N/A"),
                        #                  "score": getattr(node, 'score', None)
                        #              })
                        # ... (Add handling for other potential response types if needed)
                        else:
                             rag_result_text = str(rag_response_obj)
                             rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                             logger.warning(f"Unexpected RAG response type: {type(rag_response_obj)}")

                        logger.info(f"RAG Result {i+1} (Cleaned): {rag_result_text[:200]}...")
                        if source_nodes_info: logger.info(f"RAG Sources {i+1}: {source_nodes_info}")

                    except Exception as e:
                        logger.error(f"Error calling/processing RAG tool for issue {i+1} ('{verbatim_quote[:40]}...'): {e}", exc_info=True)
                        rag_result_text = f"Lỗi trong quá trình tra cứu RAG: {e}"
                        source_nodes_info = []
                        rag_response_successful = False

                    # --- Analyze RAG result (logic remains the same) ---
                    verification_status = "UNKNOWN"
                    verification_summary = "Không thể đưa ra kết luận xác minh từ kết quả RAG."
                    if not rag_response_successful:
                        verification_status = "RAG_CALL_FAILED"
                        verification_summary = f"**Lỗi tra cứu RAG:** Không thể gọi công cụ tra cứu. Lỗi: {rag_result_text}"
                    else:
                        # (Logic phân tích rag_result_lower, keywords, reason_lower để set status/summary giữ nguyên)
                        rag_result_lower = rag_result_text.lower()
                        negative_keywords = ["không tìm thấy", "không có", "không tồn tại", "không xác định", "không đề cập", "không quy định", "không liên quan", "không chính xác"]
                        mandatory_keywords = ["bắt buộc", "phải bao gồm", "yêu cầu", "quy định cụ thể", "phải có"]
                        not_mandatory_keywords = ["không bắt buộc", "không quy định bắt buộc", "do thỏa thuận", "không yêu cầu cụ thể"]
                        is_negative_rag = any(keyword in rag_result_lower for keyword in negative_keywords) or len(rag_result_text) < 20
                        is_mandatory_rag = any(keyword in rag_result_lower for keyword in mandatory_keywords)
                        is_not_mandatory_rag = any(keyword in rag_result_lower for keyword in not_mandatory_keywords)
                        reason_lower = reason.lower()

                        if "không chính xác" in reason_lower or "không tồn tại" in reason_lower:
                            if is_negative_rag or "không tồn tại" in rag_result_lower or "không chính xác" in rag_result_lower or "sai lệch" in rag_result_lower:
                                verification_status = "CITATION_INCORRECT_CONFIRMED"
                                verification_summary = f"**Xác minh (RAG):** Rất có khả năng trích dẫn này không tồn tại hoặc không chính xác. Phản hồi RAG: '{rag_result_text}'"
                            else:
                                verification_status = "CITATION_NEEDS_REVIEW"
                                verification_summary = f"**Thông tin tra cứu (RAG):** '{rag_result_text}'. **=> Cần người dùng đối chiếu kỹ** nội dung RAG với trích dẫn trong HĐ."
                        elif "mâu thuẫn" in reason_lower or "không nhất quán" in reason_lower or "diễn giải sai" in reason_lower or "sai luật" in reason_lower:
                            if not is_negative_rag:
                                verification_status = "CONTRADICTION_CHECK_NEEDED"
                                verification_summary = f"**Thông tin đối chiếu (RAG):** '{rag_result_text}'. **=> Cần người dùng so sánh trực tiếp** với điều khoản HĐ để xác định mâu thuẫn/sai khác."
                            else:
                                verification_status = "RAG_NEGATIVE_FOR_COMPARISON"
                                verification_summary = f"**Không tìm thấy thông tin RAG để đối chiếu:** '{rag_result_text}'"
                        elif "thiếu chi tiết" in reason_lower or "thiếu thông tin" in reason_lower:
                            if is_mandatory_rag:
                                verification_status = "MISSING_MANDATORY_CONFIRMED"
                                verification_summary = f"**Xác minh (RAG):** Luật tra cứu **có dấu hiệu yêu cầu bắt buộc** chi tiết này. Nội dung RAG: '{rag_result_text}'. **=> Nên bổ sung vào HĐ.**"
                            elif is_not_mandatory_rag:
                                verification_status = "MISSING_NOT_MANDATORY_CONFIRMED"
                                verification_summary = f"**Xác minh (RAG):** Luật tra cứu **không có dấu hiệu yêu cầu bắt buộc** chi tiết này. Nội dung RAG: '{rag_result_text}'."
                            elif not is_negative_rag:
                                verification_status = "INFO_PROVIDED_FOR_OMISSION"
                                verification_summary = f"**Thông tin đối chiếu (RAG):** '{rag_result_text}'. Cần xem xét bối cảnh để quyết định sự cần thiết."
                            else:
                                verification_status = "RAG_NEGATIVE_FOR_OMISSION"
                                verification_summary = f"**Không tìm thấy thông tin RAG về tính bắt buộc:** '{rag_result_text}'"
                        else: # Other reasons
                            if not is_negative_rag:
                                verification_status = "INFO_PROVIDED_GENERAL"
                                verification_summary = f"**Thông tin liên quan (RAG):** '{rag_result_text}'. Cần xem xét trong bối cảnh cụ thể."
                            else:
                                verification_status = "RAG_NEGATIVE_GENERAL"
                                verification_summary = f"**Không tìm thấy thông tin liên quan từ RAG:** '{rag_result_text}'"

                    # Append structured result with verbatim quote
                    verified_legal_issues_list.append({
                        "verbatim_quote": verbatim_quote, # <<< Dùng verbatim_quote
                        "initial_reason": reason,
                        "rag_query": rag_query,
                        "rag_result": rag_result_text,
                        "verification_status": verification_status,
                        "verification_summary": verification_summary,
                        "source_nodes": source_nodes_info
                    })
                 # End of RAG loop
                 workflow_step_logger.info("Finished RAG checks.")

        # Final log for this step
        workflow_step_logger.info(f"Finished step: detailed_analysis. Found {len(verified_legal_issues_list)} verified legal issues.")
        self._send_progress(ctx, "Detailed analysis checks completed.")

        return AnalysisResultsEvent(
            verified_legal_issues=verified_legal_issues_list,
            logic_issues=logic_issues_text,
            risk_issues=risk_issues_text
        )

    @step()
    async def synthesize_output(self, ctx: Context, ev: AnalysisResultsEvent) -> StopEvent:
        """Synthesizes the final report and extracts annotations using verbatim quotes."""
        workflow_step_logger.info("Starting step: synthesize_output")
        self._send_progress(ctx, "Synthesizing final report...")

        # Retrieve context
        contract_type = await ctx.get("contract_type", "Không xác định")
        key_elements_list_ctx = await ctx.get("key_elements", [])
        cited_laws_list_ctx = await ctx.get("cited_laws", [])
        query = await ctx.get("query", config.DEFAULT_QUERY)

        context_info = {
            "contract_type": contract_type,
            "key_elements": key_elements_list_ctx,
            "cited_laws": cited_laws_list_ctx
        }

        # Parse logic and risk findings (expecting 'exact_quote')
        logic_findings_data = extract_json_from_response(ev.logic_issues)
        risk_findings_data = extract_json_from_response(ev.risk_issues)
        logic_findings_list = logic_findings_data.get("logic_findings", [])
        risk_findings_list = risk_findings_data.get("risk_findings", [])
        verified_legal_issues_list = ev.verified_legal_issues

        workflow_step_logger.info(f"Data for synthesis: Legal={len(verified_legal_issues_list)}, Logic={len(logic_findings_list)}, Risk={len(risk_findings_list)}")

        # --- Create Report Sections using Python ---
        report_sections = {"legal": "", "logic": "", "risk": ""}
        annotations: List[Dict[str, Any]] = []

        # --- Section II: Legal Analysis ---
        legal_details_lines = ["### 2.1. Phân tích Pháp lý (Đối chiếu với Luật)", ""]
        if not verified_legal_issues_list:
            legal_details_lines.append("*Không có điểm pháp lý cụ thể nào được xác định cần kiểm tra thêm.*")
        else:
            for i, issue in enumerate(verified_legal_issues_list):
                # <<< CHANGE: Use verbatim_quote for display and annotation >>>
                verbatim_quote = issue.get('verbatim_quote', 'N/A')
                reason = issue.get('initial_reason', 'N/A')
                reason = re.sub(r"<think>.*?</think>\s*", "", reason, flags=re.DOTALL).strip()
                summary = issue.get('verification_summary', 'N/A')
                
                print("=======================================")
                print(repr(summary))
                print("=======================================")
                summary = re.sub(r'\n{2,}', '\n', summary).strip()  # Clean up multiple newlines
                
                status = issue.get('verification_status', 'UNKNOWN')

                # Use verbatim_quote for the report display
                legal_details_lines.append(f"- **{i+1}. Điểm HĐ cần lưu ý:** `{verbatim_quote}`")
                legal_details_lines.append(f"    - **Lý do kiểm tra (LLM gợi ý):** {reason}")
                legal_details_lines.append(f"    - **Kết quả Xác minh (RAG - Status: {status}):** {summary}")

                sources_str = ""
                if issue.get('source_nodes'):
                     sources = [f"{s.get('file_name', '?').split('/')[-1]}:Trang {s.get('page', '?')}" for s in issue.get('source_nodes', [])]
                     if sources: sources_str = "; ".join(sources)
                if sources_str:
                    legal_details_lines.append(f"    - *Nguồn tham khảo (RAG):* {sources_str}")
                legal_details_lines.append("---")

                # <<< CHANGE: Use verbatim_quote for location_hint >>>
                annotations.append({
                    "type": "legal",
                    "status": status,
                    "location_hint": verbatim_quote, # Use the exact quote for UI highlight
                    "summary": reason,
                    "details": summary
                })
        report_sections["legal"] = "\n".join(legal_details_lines)

        # --- Section III: Logic Analysis ---
        logic_details_lines = ["### 2.2. Phân tích Logic & Ngữ nghĩa (Vấn đề nội tại)", ""]
        if not logic_findings_list:
            logic_details_lines.append("*Không phát hiện vấn đề logic hay ngữ nghĩa nào.*")
        else:
            for finding in logic_findings_list:
                finding_type = finding.get('type', 'N/A')
                description = finding.get('description', 'N/A')
                # <<< CHANGE: Prioritize 'exact_quote' for location hint >>>
                exact_quotes = finding.get('exact_quote', []) # Expecting a list now
                location_hint = exact_quotes[0] if exact_quotes and isinstance(exact_quotes, list) else description # Default to first quote or description

                logic_details_lines.append(f"- **Loại:** `{finding_type}`")
                logic_details_lines.append(f"  - **Mô tả:** {description}")
                # Display quotes if available, otherwise fallback
                if exact_quotes and isinstance(exact_quotes, list):
                    logic_details_lines.append(f"  - **Trích dẫn liên quan:**")
                    for q in exact_quotes: logic_details_lines.append(f"    - `{q}`")
                elif finding.get('clauses_involved'): # Fallback to clauses_involved if no exact_quote
                     logic_details_lines.append(f"  - **Điều khoản liên quan (ước tính):** {json.dumps(finding.get('clauses_involved'), ensure_ascii=False)}")
                elif finding.get('inconsistent_terms'):
                     logic_details_lines.append(f"  - **Thuật ngữ không nhất quán:** {json.dumps(finding.get('inconsistent_terms'), ensure_ascii=False)}")
                elif finding.get('term'):
                     logic_details_lines.append(f"  - **Thuật ngữ không định nghĩa:** {finding.get('term')}")


                annotations.append({
                    "type": "logic",
                    "status": finding_type,
                    "location_hint": location_hint, # Use extracted/prioritized hint
                    "summary": description,
                    "details": f"Loại: {finding_type}. Mô tả: {description}"
                })
        report_sections["logic"] = "\n".join(logic_details_lines)

        # --- Section IV: Risk Analysis ---
        risk_details_lines = ["### 2.3. Phân tích Rủi ro", ""]
        if not risk_findings_list:
            risk_details_lines.append("*Không phát hiện rủi ro cụ thể nào dựa trên các tiêu chí phân tích.*")
        else:
            for finding in risk_findings_list:
                risk_type = finding.get('risk_type', 'N/A')
                description = finding.get('description', 'N/A')
                suggestion = finding.get('suggestion', 'N/A')
                # <<< CHANGE: Prioritize 'exact_quote' for location hint >>>
                exact_quote = finding.get('exact_quote') # Expecting a string
                clause_involved = finding.get('clause_involved') # Fallback
                location_hint = exact_quote if exact_quote else (clause_involved if clause_involved else description)

                risk_details_lines.append(f"- **Loại rủi ro:** `{risk_type}`")
                risk_details_lines.append(f"  - **Mô tả:** {description}")
                # Display quote if available
                if exact_quote:
                    risk_details_lines.append(f"  - **Trích dẫn liên quan:** `{exact_quote}`")
                elif clause_involved: # Fallback display
                     risk_details_lines.append(f"  - **Điều khoản liên quan (ước tính):** {clause_involved}")

                risk_details_lines.append(f"  - **Đề xuất:** {suggestion}")

                annotations.append({
                    "type": "risk",
                    "status": risk_type,
                    "location_hint": location_hint, # Use extracted/prioritized hint
                    "summary": description,
                    "details": suggestion
                })
        report_sections["risk"] = "\n".join(risk_details_lines)

        # --- Prepare input for Synthesis LLM (Summary & Recs only) ---
        # (Logic to prepare num_legal, num_logic, num_risk, highlights, suggestions remains the same)
        num_legal = len(verified_legal_issues_list)
        num_logic = len(logic_findings_list)
        num_risk = len(risk_findings_list)

        legal_highlights = ""
        actionable_legal_suggestions = []
        if num_legal > 0:
             confirmed_issues = [iss for iss in verified_legal_issues_list if "CONFIRMED" in iss.get("verification_status", "")]
             needs_review_issues = [iss for iss in verified_legal_issues_list if "NEEDS_REVIEW" in iss.get("verification_status", "")]
             if confirmed_issues:
                 first_confirmed = confirmed_issues[0]
                 legal_highlights = f", ví dụ: {first_confirmed.get('initial_reason', '')[:50]}... ({first_confirmed.get('verification_status', '')})"
                 if "=>" in first_confirmed.get('verification_summary', ''):
                     actionable_legal_suggestions.append(first_confirmed.get('verification_summary').split("=>")[-1].strip())
             elif needs_review_issues:
                  first_review = needs_review_issues[0]
                  legal_highlights = f", ví dụ: {first_review.get('initial_reason', '')[:50]}... ({first_review.get('verification_status', '')})"
             else:
                  first_issue = verified_legal_issues_list[0]
                  legal_highlights = f", ví dụ: {first_issue.get('initial_reason', '')[:50]}... ({first_issue.get('verification_status', '')})"
                  if "=>" in first_issue.get('verification_summary', ''):
                       actionable_legal_suggestions.append(first_issue.get('verification_summary').split("=>")[-1].strip())

        logic_highlights = f", ví dụ: {logic_findings_list[0].get('description', '')[:50]}..." if num_logic > 0 else ""
        risk_highlights = f", ví dụ: {risk_findings_list[0].get('description', '')[:50]}..." if num_risk > 0 else ""
        all_suggestions = [finding.get('suggestion') for finding in risk_findings_list if finding.get('suggestion')]
        all_suggestions.extend(actionable_legal_suggestions)

        # --- Get Synthesis Prompt from Store ---
        synthesis_prompt = self.prompt_store.get_synthesis_prompt(
            context_info=context_info, query=query, num_legal=num_legal, num_logic=num_logic, num_risk=num_risk,
            legal_highlights=legal_highlights, logic_highlights=logic_highlights, risk_highlights=risk_highlights,
            suggestions=all_suggestions
        )

        # --- Call Synthesis LLM ---
        workflow_step_logger.info("Calling LLM for Synthesis (Summary & Recommendations Only)...")
        try:
            synthesis_output = await Settings.llm.acomplete(synthesis_prompt)
            synthesis_text = re.sub(r"<think>.*?</think>\s*", "", synthesis_output.text, flags=re.DOTALL).strip()
            workflow_step_logger.info("Synthesis LLM call completed.")
            logger.debug(f"Synthesis LLM raw response:\n{synthesis_text}")
        except Exception as e:
            logger.error(f"Error calling LLM for synthesis: {e}", exc_info=True)
            synthesis_text = "**I. Tóm tắt:**\n(Lỗi: Không thể tạo tóm tắt tự động.)\n\n**V. Tổng hợp Đề xuất:**\n(Lỗi: Không thể tạo đề xuất tự động.)"

        # --- Combine parts to create the final report ---
        summary_match = re.search(r"\*\*I\. Tóm tắt:\*\*(.*?)(?=\n\*\*V\.|\Z)", synthesis_text, re.DOTALL)
        recs_match = re.search(r"\*\*V\. Tổng hợp Đề xuất:\*\*(.*)", synthesis_text, re.DOTALL)
        final_summary = summary_match.group(1).strip() if summary_match else "(Lỗi: Không trích xuất được Tóm tắt)"
        final_recs = recs_match.group(1).strip() if recs_match else "(Lỗi: Không trích xuất được Đề xuất)"

        # Construct final report string using updated Markdown structure
        final_report_parts = [
            "# Báo cáo Phân tích Hợp đồng", # H1 Title
            "---",
            "## 1. Tóm tắt", # H2 Section
            final_summary,
            "---",
            "## 2. Phân tích Chi tiết", # H2 Section
            report_sections["legal"],       # Content includes H3 subsections
            report_sections["logic"],       # Content includes H3 subsections
            report_sections["risk"],        # Content includes H3 subsections
            "---",
            "## 3. Tổng hợp Đề xuất", # H2 Section
            final_recs,
            "---",
            "*Lưu ý: Báo cáo này được tạo tự động và chỉ mang tính tham khảo. Cần có sự xem xét của chuyên gia pháp lý.*"
        ]
        final_report = "\n\n".join(final_report_parts)

        workflow_step_logger.info("Final report synthesized.")
        logger.debug(f"Final Report (excerpt):\n{final_report[:500]}...")
        workflow_step_logger.info(f"Extracted {len(annotations)} annotations for UI.")
        logger.debug(f"Annotations sample: {annotations[:2]}")

        self._send_progress(ctx, "Synthesis completed.")
        return StopEvent(result=FinalOutputEvent(report=final_report, annotations=annotations))

# --- Initialization Function (Giữ nguyên) ---
async def initialize_analysis_system() -> Optional[BaseTool]:
    # (Giữ nguyên code hàm này)
    logger.info("Initializing LlamaIndex Settings and Qdrant connection...")
    try:
        Settings.llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, request_timeout=config.REQUEST_TIMEOUT)
        Settings.embed_model = OllamaEmbedding(model_name=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL, request_timeout=config.REQUEST_TIMEOUT)
        logger.info(f"LLM ({config.LLM_MODEL}) and Embeddings ({config.EMBED_MODEL}) configured.")
        try:
            async_client = AsyncQdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=10)
            await async_client.get_collections()
            logger.info("Successfully connected to Qdrant.")
        except Exception as q_err:
            logger.error(f"Failed to connect to Qdrant: {q_err}", exc_info=True)
            raise ConnectionError(f"Could not connect to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}.") from q_err
        vector_store = QdrantVectorStore(aclient=async_client, collection_name=config.QDRANT_COLLECTION)
        logger.info(f"QdrantVectorStore initialized for collection '{config.QDRANT_COLLECTION}'.")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("VectorStoreIndex created from Qdrant store.")
        query_engine = index.as_query_engine(similarity_top_k=config.SIMILARITY_TOP_K)
        logger.info(f"Query engine created with similarity_top_k={config.SIMILARITY_TOP_K}.")
        law_tool = QueryEngineTool.from_defaults(query_engine, name=config.RAG_TOOL_NAME, description=config.RAG_TOOL_DESC)
        logger.info(f"RAG QueryEngineTool '{config.RAG_TOOL_NAME}' created.")
        logger.info("Initialization complete.")
        return law_tool
    except Exception as e:
        logger.error(f"Failed to initialize analysis system: {e}", exc_info=True)
        return None
# workflow.py
import logging
import json
import asyncio
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable, Tuple

# LlamaIndex Imports
from llama_index.core import Settings, VectorStoreIndex # <<< Import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.tools import QueryEngineTool, BaseTool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
)
# Qdrant Client
from qdrant_client import AsyncQdrantClient

# Project Imports
from prompts import PromptStore
from utils import extract_json_from_response, normalize_law_name_simple
import config

# Configure logging (giữ nguyên)
logger = logging.getLogger(__name__)
workflow_step_logger = logging.getLogger("WorkflowSteps")
# ... (phần logging khác giữ nguyên)

# --- Event Definitions ---

class WorkflowStartEvent(StartEvent):
    contract_text: str
    query: str
    tools: List[BaseTool]
    index: VectorStoreIndex # <<< Thêm index vào StartEvent

# ... (Các Event Definitions khác giữ nguyên)
class ContractTextEvent(Event):
    contract_text: str

class ContextAnalyzedEvent(Event):
    contract_type: Optional[str]
    key_elements: List[str]
    cited_legal_documents: List[str]
    contract_text: str

class AnalysisResultsEvent(Event):
    verified_legal_issues: List[Dict[str, Any]]
    logic_issues: str
    risk_issues: str

class FinalOutputEvent(Event):
    report: str
    annotations: List[Dict[str, Any]]

class ProgressEvent(Event):
    progress: str
# --- Workflow Definition ---

class MultiAgentContractReviewWorkflow(Workflow):
    # ... ( __init__ và _send_progress giữ nguyên)
    def __init__(
        self,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.progress_callback = progress_callback
        self.prompt_store = PromptStore()

    def _send_progress(self, ctx: Context, message: str):
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

    def _parse_law_citation_for_filters(self, text_to_parse: str) -> Dict[str, str]:
        # (Giữ nguyên logic hàm này)
        filters = {}
        if not text_to_parse:
            return filters
        text_lower = text_to_parse.lower()
        article_search_match = re.search(r"(?:điều|article|art\.|art)[\s\.số]*(\d+)", text_lower)
        if article_search_match:
            filters["article_number"] = article_search_match.group(1)
        law_phrase_patterns = [
            r"((?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quyết\s+định|pháp\s+lệnh|hiến\s+pháp)[\s\w\/\(\)\.,-]+(?:năm\s+\d{4}|\b\d{4}\b))",
            r"((?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|quyết\s+định|pháp\s+lệnh|hiến\s+pháp)\s+[\w\s\/\(\)\.,-]+)"
        ]
        extracted_law_phrase_for_normalization = None
        for pattern_idx, law_pattern_regex in enumerate(law_phrase_patterns):
            law_match = re.search(law_pattern_regex, text_lower)
            if law_match:
                extracted_law_phrase_for_normalization = law_match.group(1).strip()
                workflow_step_logger.debug(f"Law phrase pattern {pattern_idx} matched: '{extracted_law_phrase_for_normalization}'")
                break
        if extracted_law_phrase_for_normalization:
            normalized_law_name = normalize_law_name_simple(extracted_law_phrase_for_normalization)
            if normalized_law_name:
                filters["law_normalized_simple"] = normalized_law_name
                workflow_step_logger.info(
                    f"Extracted law for filtering: Original phrase='{extracted_law_phrase_for_normalization}', "
                    f"Normalized='{normalized_law_name}', Article='{filters.get('article_number')}'"
                )
            else:
                workflow_step_logger.warning(f"Could not normalize extracted law phrase: '{extracted_law_phrase_for_normalization}'")
        if "article_number" in filters and "law_normalized_simple" not in filters:
            workflow_step_logger.info(f"Found article '{filters['article_number']}' but no law name for filtering, removing article from filters.")
            filters.pop("article_number")
        return {k: v for k, v in filters.items() if v}

    @step()
    async def preprocess_input(self, ctx: Context, ev: WorkflowStartEvent) -> ContractTextEvent:
        workflow_step_logger.info("Starting step: preprocess_input")
        await ctx.set("query", ev.query)
        await ctx.set("tools", ev.tools)
        await ctx.set("index", ev.index) # <<< Lưu index vào context
        await ctx.set("contract_text", ev.contract_text)
        self._send_progress(ctx, "Preprocessing completed.")
        return ContractTextEvent(contract_text=ev.contract_text)

    @step()
    async def analyze_context(self, ctx: Context, ev: ContractTextEvent) -> ContextAnalyzedEvent:
        # (Giữ nguyên logic hàm này)
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
        workflow_step_logger.info("Starting step: detailed_analysis")
        contract_text = ev.contract_text
        query = await ctx.get("query")
        tools: List[BaseTool] = await ctx.get("tools")
        index: VectorStoreIndex = await ctx.get("index") # <<< Lấy index từ context
        contract_type = ev.contract_type
        cited_laws_list = ev.cited_legal_documents

        # ... (Phần gọi LLM cho logic, risk, legal_review giữ nguyên)
        logic_prompt = self.prompt_store.get_logic_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        risk_prompt = self.prompt_store.get_risk_analysis_prompt(
            contract_type=contract_type, query=query, contract_text=contract_text
        )
        legal_prompt = self.prompt_store.get_legal_review_prompt(
            contract_text=contract_text, cited_laws_list=cited_laws_list
        )

        self._send_progress(ctx, "Running detailed analysis (Logic, Risk, Legal Red Flags)...")
        try:
            logic_response, risk_response, legal_review_response = await asyncio.gather(
                Settings.llm.acomplete(logic_prompt),
                Settings.llm.acomplete(risk_prompt),
                Settings.llm.acomplete(legal_prompt),
                return_exceptions=True
            )

            default_logic_error = '{"logic_findings": [{"type": "error", "description": "Lỗi gọi LLM phân tích Logic", "exact_quote": []}]}'
            default_risk_error = '{"risk_findings": [{"risk_type": "error", "description": "Lỗi gọi LLM phân tích Rủi ro", "exact_quote": "", "suggestion": "N/A"}]}'
            default_legal_error = "Lỗi gọi LLM xác định điểm pháp lý ban đầu."

            logic_issues_text = logic_response.text if not isinstance(logic_response, Exception) else default_logic_error
            if isinstance(logic_response, Exception): logger.error(f"Error in Logic Analysis LLM call: {logic_response}", exc_info=logic_response)

            risk_issues_text = risk_response.text if not isinstance(risk_response, Exception) else default_risk_error
            if isinstance(risk_response, Exception): logger.error(f"Error in Risk Analysis LLM call: {risk_response}", exc_info=risk_response)

            legal_review_text = legal_review_response.text.strip() if not isinstance(legal_review_response, Exception) else default_legal_error
            if isinstance(legal_review_response, Exception): logger.error(f"Error in Legal Red Flag LLM call: {legal_review_response}", exc_info=legal_review_response)

            workflow_step_logger.info("Detailed analysis LLM calls completed.")
            logger.debug(f"Logic analysis response: {logic_issues_text[:200]}...")
            logger.debug(f"Risk analysis response: {risk_issues_text[:200]}...")
            logger.debug(f"Initial legal review response:\n{legal_review_text}")

        except Exception as e:
            logger.error(f"Critical error during concurrent LLM calls: {e}", exc_info=True)
            logic_issues_text = default_logic_error
            risk_issues_text = default_risk_error
            legal_review_text = "Lỗi hệ thống khi gọi LLM."

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
            rag_tool: Optional[BaseTool] = next((tool for tool in tools if hasattr(tool, 'metadata') and tool.metadata.name == config.RAG_TOOL_NAME), None)

            if not rag_tool:
                logger.error(f"RAG tool '{config.RAG_TOOL_NAME}' not found.")
                verified_legal_issues_list.append({
                    "verbatim_quote": "Lỗi Hệ thống", "initial_reason": "Công cụ RAG lỗi.",
                    "rag_query": "N/A", "rag_result": "N/A", "verification_status": "ERROR",
                    "verification_summary": "Lỗi cấu hình RAG.", "source_nodes": []
                })
            else:
                workflow_step_logger.info(f"Using RAG tool: {rag_tool.metadata.name}")
                # ... (Phần parse potential_issues giữ nguyên)
                potential_issues = []
                try:
                    pattern = re.compile(
                    r"-\s*\*\*Vị trí/Trích dẫn Nguyên văn:\*\*\s*(.*?)\s*-?\s*\*\*Lý do cần kiểm tra:\*\*\s*(.*?)(?=\n-\s*\*\*Vị trí/Trích dẫn Nguyên văn:\*\*|\Z)",
                    re.DOTALL | re.IGNORECASE
                    )
                    matches = pattern.findall(legal_review_text)
                    workflow_step_logger.info(f"Regex matches found for legal issues: {len(matches)}")
                    for match_item in matches:
                        quote = match_item[0].strip()
                        reason_llm_suggested_item = match_item[1].strip().replace(no_issues_found_msg, '').strip()
                        if quote and reason_llm_suggested_item:
                            potential_issues.append({
                                "verbatim_quote": quote,
                                "reason_llm_suggested": reason_llm_suggested_item
                            })
                    if not matches and legal_review_text != no_issues_found_msg:
                        logger.warning("Could not parse legal red flags using updated regex. Check LLM output format.")
                        potential_issues.append({ "verbatim_quote": "Lỗi phân tích: Không tìm thấy định dạng Markdown mong muốn.", "reason_llm_suggested": legal_review_text })
                except Exception as parse_err:
                    logger.error(f"Error parsing legal review text: {parse_err}", exc_info=True)
                    verified_legal_issues_list.append({
                        "verbatim_quote": "Lỗi Phân tích Cú pháp", "initial_reason": f"Lỗi parse: {parse_err}",
                        "rag_query": "N/A", "rag_result": legal_review_text, "verification_status": "ERROR",
                        "verification_summary": "Không thể phân tích các điểm pháp lý từ LLM.", "source_nodes": []
                    })
                    potential_issues = []

                workflow_step_logger.info(f"Potential legal issues identified for RAG check: {len(potential_issues)}")
                self._send_progress(ctx, f"Found {len(potential_issues)} potential legal issues. Starting RAG checks...")

                for i, issue in enumerate(potential_issues):
                    verbatim_quote = issue.get("verbatim_quote", "N/A")
                    reason_llm_suggested = issue.get("reason_llm_suggested", "N/A")
                    self._send_progress(ctx, f"Checking legal issue {i+1}/{len(potential_issues)}: '{verbatim_quote[:40]}...'")

                    rag_query = ""
                    rag_result_text = "Lỗi tra cứu RAG."
                    source_nodes_info = [] # Reset for each issue
                    rag_response_successful = False

                    parsed_filter_criteria_dict = self._parse_law_citation_for_filters(reason_llm_suggested)
                    active_metadata_filters: Optional[MetadataFilters] = None
                    if parsed_filter_criteria_dict.get("law_normalized_simple") and parsed_filter_criteria_dict.get("article_number"): # Giữ nguyên logic AND
                        active_metadata_filters = MetadataFilters(filters=[
                            ExactMatchFilter(key="law_normalized_simple", value=parsed_filter_criteria_dict["law_normalized_simple"]),
                            ExactMatchFilter(key="article_number", value=parsed_filter_criteria_dict["article_number"])
                        ])
                        workflow_step_logger.info(f"✅ Applying RAG metadata filters: {parsed_filter_criteria_dict}")
                    else:
                        workflow_step_logger.info(f"⚠️ Skip filter (not enough data or only partial data): {parsed_filter_criteria_dict}")

                    issue_type, verification_instruction = self.categorize_legal_issue(reason_llm_suggested)
                    if not verification_instruction:
                        verification_instruction = "Cung cấp thông tin pháp lý liên quan nhất cho đoạn văn bản sau."
                        issue_type = "general_compliance"
                    rag_query = f"Đoạn hợp đồng cần phân tích: '{verbatim_quote}'. Yêu cầu kiểm tra: {verification_instruction}"
                    workflow_step_logger.info(f"Categorized Issue Type: {issue_type}")
                    workflow_step_logger.info(f"RAG Query {i+1}: {rag_query}")

                    if active_metadata_filters:
                        try:
                            workflow_step_logger.info(f"Attempting RAG with filters: {active_metadata_filters}")
                            filtered_query_engine = index.as_query_engine(
                                filters=active_metadata_filters,
                                similarity_top_k=config.SIMILARITY_TOP_K
                            )
                            # LlamaIndex Response object
                            response_obj_filtered = await filtered_query_engine.aquery(rag_query)
                            rag_response_successful = True

                            rag_result_text = response_obj_filtered.response
                            rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                            
                            source_nodes_info = [] # Reset
                            for node in response_obj_filtered.source_nodes:
                                metadata = node.node.metadata
                                source_nodes_info.append({
                                    "file_name": metadata.get("law_full_name", metadata.get("file_name", "N/A")),
                                    "page": metadata.get("article_file_name", metadata.get("page_label", "N/A")),
                                    "score": node.score
                                })
                            logger.info(f"RAG Result (filtered) {i+1}: {rag_result_text[:200]}...")
                            if source_nodes_info: logger.info(f"RAG Sources (filtered) {i+1}: {source_nodes_info}")

                        except Exception as e_filter:
                            logger.error(f"Error during filtered RAG query for issue {i+1} ('{verbatim_quote[:40]}...'): {e_filter}", exc_info=True)
                            rag_result_text = f"Lỗi trong quá trình tra cứu RAG (có filter): {e_filter}"
                            source_nodes_info = []
                            rag_response_successful = False
                    else: # No active filters, use the general RAG tool
                        try:
                            # rag_tool.acall expects a string or dict like {"input": "query_string"}
                            # If rag_tool is QueryEngineTool, it internally uses query_engine.aquery(input_str)
                            # or query_engine.aquery(QueryBundle(input_str))
                            rag_response_obj = await rag_tool.acall(rag_query) # Pass string directly
                            rag_response_successful = True

                            # QueryEngineTool.acall returns a ToolOutput object.
                            # ToolOutput.content is the string response.
                            # ToolOutput.raw_output is the original Response object from the query engine.
                            if hasattr(rag_response_obj, 'raw_output') and rag_response_obj.raw_output:
                                rag_result_text = rag_response_obj.content # String response
                                rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                                
                                source_nodes_info = [] # Reset
                                # Access source_nodes from the raw_output (which should be a LlamaIndex Response)
                                for node in rag_response_obj.raw_output.source_nodes:
                                    metadata = node.node.metadata
                                    source_nodes_info.append({
                                        "file_name": metadata.get("law_full_name", metadata.get("file_name", "N/A")),
                                        "page": metadata.get("article_file_name", metadata.get("page_label", metadata.get("document_title", "N/A"))),
                                        "score": node.score
                                    })
                            else:
                                rag_result_text = str(rag_response_obj) # Fallback
                                rag_result_text = re.sub(r"<think>.*?</think>\s*", "", rag_result_text, flags=re.DOTALL).strip()
                                logger.warning(f"Unexpected RAG response type (no filter, no raw_output): {type(rag_response_obj)}. Processed as string.")
                                source_nodes_info = []

                            logger.info(f"RAG Result (no filter) {i+1}: {rag_result_text[:200]}...")
                            if source_nodes_info: logger.info(f"RAG Sources (no filter) {i+1}: {source_nodes_info}")
                        except Exception as e_no_filter:
                            logger.error(f"Error calling/processing RAG tool (no filter) for issue {i+1} ('{verbatim_quote[:40]}...'): {e_no_filter}", exc_info=True)
                            rag_result_text = f"Lỗi trong quá trình tra cứu RAG (không filter): {e_no_filter}"
                            source_nodes_info = []
                            rag_response_successful = False

                    # --- Analyze RAG result (logic remains the same) ---
                    verification_status, verification_summary = "UNKNOWN", "Không thể đưa ra kết luận xác minh từ kết quả RAG."
                    if not rag_response_successful:
                        verification_status = "RAG_CALL_FAILED"
                        verification_summary = f"**Lỗi tra cứu RAG:** Không thể gọi công cụ tra cứu. Lỗi: {rag_result_text}"
                    else:
                        verification_status, verification_summary = self.analyze_rag_result_for_issue_type(
                            issue_type, rag_result_text, verbatim_quote, reason_llm_suggested
                        )

                    verified_legal_issues_list.append({
                        "verbatim_quote": verbatim_quote,
                        "initial_reason": reason_llm_suggested,
                        "classified_issue_type": issue_type,
                        "rag_query": rag_query,
                        "rag_filters_applied": parsed_filter_criteria_dict if active_metadata_filters else None,
                        "rag_result": rag_result_text,
                        "verification_status": verification_status,
                        "verification_summary": verification_summary,
                        "source_nodes": source_nodes_info
                    })
                workflow_step_logger.info("Finished RAG checks.")
        # ... (Phần còn lại của detailed_analysis và các hàm helper giữ nguyên)
        workflow_step_logger.info(f"Finished step: detailed_analysis. Found {len(verified_legal_issues_list)} verified legal issues.")
        self._send_progress(ctx, "Detailed analysis checks completed.")
        return AnalysisResultsEvent(
            verified_legal_issues=verified_legal_issues_list,
            logic_issues=logic_issues_text,
            risk_issues=risk_issues_text
        )

    def categorize_legal_issue(self, reason_llm_suggested: str) -> Tuple[str, str]:
        # (Giữ nguyên logic hàm này)
        reason_lower = reason_llm_suggested.lower()
        if "không chính xác" in reason_lower or "không tồn tại" in reason_lower or "sai lệch" in reason_lower or "trích dẫn sai" in reason_lower:
            return "citation_accuracy", "Xác minh tính chính xác, sự tồn tại và nội dung của các văn bản pháp luật hoặc thông tin được trích dẫn/tham chiếu trong đoạn văn bản này."
        elif "mâu thuẫn" in reason_lower or "không nhất quán" in reason_lower or "diễn giải sai" in reason_lower or "trái luật" in reason_lower or "sai luật" in reason_lower:
            return "contradiction_compliance", "Đối chiếu đoạn văn bản này với các quy định pháp luật liên quan để xác định xem có sự mâu thuẫn, không nhất quán, diễn giải sai hoặc vi phạm pháp luật nào không. Cung cấp các điều luật cụ thể để so sánh."
        elif "thiếu chi tiết" in reason_lower or "thiếu thông tin" in reason_lower or "bỏ sót" in reason_lower or "không đề cập" in reason_lower:
            law_match = re.search(r"(luật|bộ luật|nghị định|thông tư)[\s\w\/\-]+", reason_lower)
            law_ref = law_match.group(0).strip() if law_match else "pháp luật liên quan"
            return "omission_completeness", f"Kiểm tra xem {law_ref} có quy định bắt buộc nào liên quan đến nội dung hoặc các yếu tố trong đoạn văn bản này mà có thể đã bị bỏ sót hoặc chưa được đề cập đầy đủ không. Nếu có, nêu rõ quy định đó."
        elif "rủi ro" in reason_lower or "cảnh báo" in reason_lower or "lưu ý" in reason_lower:
            return "potential_risk_legal_basis", "Phân tích các rủi ro pháp lý tiềm ẩn liên quan đến nội dung của đoạn văn bản này và cung cấp cơ sở pháp lý (nếu có) cho các rủi ro đó."
        else:
            return "general_review", "Cung cấp thông tin pháp lý tổng quan và các điểm cần lưu ý (nếu có) liên quan đến đoạn văn bản này theo quy định của pháp luật hiện hành."

    def analyze_rag_result_for_issue_type(
        self, issue_type: str, rag_result_text: str, verbatim_quote: str, reason_llm_suggested: str
    ) -> Tuple[str, str]:
        # (Giữ nguyên logic hàm này)
        rag_result_lower = rag_result_text.lower()
        status = "UNKNOWN"
        summary = f"**Thông tin tra cứu (RAG):** '{rag_result_text}'. Cần người dùng xem xét dựa trên loại nghi vấn: {issue_type}."
        negative_keywords = ["không tìm thấy", "không có", "không tồn tại", "không xác định", "không đề cập", "không quy định", "không liên quan", "không chính xác"]
        mandatory_keywords = ["bắt buộc", "phải bao gồm", "yêu cầu", "quy định cụ thể", "phải có"]
        not_mandatory_keywords = ["không bắt buộc", "không quy định bắt buộc", "do thỏa thuận", "không yêu cầu cụ thể", "tùy nghi"]
        conflicting_keywords = ["mâu thuẫn với", "trái với", "không phù hợp với", "vi phạm"]
        is_negative_rag = any(keyword in rag_result_lower for keyword in negative_keywords) or len(rag_result_text) < 30
        is_mandatory_rag = any(keyword in rag_result_lower for keyword in mandatory_keywords)
        is_not_mandatory_rag = any(keyword in rag_result_lower for keyword in not_mandatory_keywords)
        is_conflicting_rag = any(keyword in rag_result_lower for keyword in conflicting_keywords)

        if issue_type == "citation_accuracy":
            if is_negative_rag or "không chính xác" in rag_result_lower or "sai lệch" in rag_result_lower:
                status = "CITATION_INCORRECT_CONFIRMED"
                summary = f"**Xác minh (RAG):** Có dấu hiệu trích dẫn/tham chiếu trong hợp đồng không tồn tại, không chính xác hoặc sai lệch. Phản hồi RAG: '{rag_result_text}'"
            elif ("khớp" in rag_result_lower or "chính xác" in rag_result_lower or "tồn tại" in rag_result_lower or "đúng" in rag_result_lower) and not is_negative_rag :
                status = "CITATION_CORRECT_CONFIRMED"
                summary = f"**Xác minh (RAG):** Trích dẫn/tham chiếu có vẻ chính xác dựa trên thông tin RAG. RAG: '{rag_result_text}'"
            else:
                status = "CITATION_NEEDS_REVIEW"
                summary = f"**Thông tin tra cứu (RAG):** '{rag_result_text}'. **=> Cần người dùng đối chiếu kỹ** nội dung RAG với trích dẫn trong HĐ."
        elif issue_type == "contradiction_compliance":
            if is_conflicting_rag:
                status = "CONTRADICTION_POTENTIAL_CONFIRMED"
                summary = f"**Xác minh (RAG):** Có dấu hiệu nội dung hợp đồng mâu thuẫn hoặc không tuân thủ quy định pháp luật. RAG: '{rag_result_text}'. **=> Cần người dùng xem xét kỹ và đối chiếu.**"
            elif not is_negative_rag:
                status = "COMPLIANCE_CHECK_DATA_PROVIDED"
                summary = f"**Thông tin đối chiếu (RAG):** '{rag_result_text}'. **=> Cần người dùng so sánh trực tiếp** với điều khoản HĐ để xác định mâu thuẫn/sai khác hoặc tính tuân thủ."
            else:
                status = "RAG_NEGATIVE_FOR_COMPARISON"
                summary = f"**Không tìm thấy thông tin RAG cụ thể để đối chiếu về mâu thuẫn/tuân thủ:** '{rag_result_text}'"
        elif issue_type == "omission_completeness":
            if is_mandatory_rag:
                status = "OMISSION_MANDATORY_CONTENT_IDENTIFIED"
                summary = f"**Xác minh (RAG):** Pháp luật có dấu hiệu yêu cầu bắt buộc các nội dung/chi tiết liên quan (có thể đang thiếu trong HĐ). RAG: '{rag_result_text}'. **=> Nên xem xét bổ sung.**"
            elif is_not_mandatory_rag:
                status = "OMISSION_NOT_MANDATORY_CONFIRMED"
                summary = f"**Xác minh (RAG):** Pháp luật không có dấu hiệu yêu cầu bắt buộc các chi tiết này. RAG: '{rag_result_text}'."
            elif not is_negative_rag:
                status = "INFO_PROVIDED_FOR_OMISSION_CHECK"
                summary = f"**Thông tin đối chiếu (RAG):** '{rag_result_text}'. Cần xem xét bối cảnh để quyết định sự cần thiết của việc bổ sung thông tin vào HĐ."
            else:
                status = "RAG_NEGATIVE_FOR_OMISSION_CHECK"
                summary = f"**Không tìm thấy thông tin RAG về tính bắt buộc của nội dung liên quan:** '{rag_result_text}'"
        elif issue_type == "potential_risk_legal_basis" or issue_type == "general_review":
            if not is_negative_rag:
                status = "INFO_PROVIDED_GENERAL"
                summary = f"**Thông tin liên quan (RAG):** '{rag_result_text}'. Cần xem xét trong bối cảnh cụ thể của hợp đồng và mục tiêu phân tích."
            else:
                status = "RAG_NEGATIVE_GENERAL"
                summary = f"**Không tìm thấy thông tin pháp lý cụ thể liên quan từ RAG:** '{rag_result_text}'"
        return status, summary

    @step()
    async def synthesize_output(self, ctx: Context, ev: AnalysisResultsEvent) -> StopEvent:
        # (Giữ nguyên logic hàm này, chỉ đảm bảo các trường dữ liệu được truyền đúng)
        workflow_step_logger.info("Starting step: synthesize_output")
        self._send_progress(ctx, "Synthesizing final report...")

        contract_type = await ctx.get("contract_type", "Không xác định")
        key_elements_list_ctx = await ctx.get("key_elements", [])
        cited_laws_list_ctx = await ctx.get("cited_laws", [])
        query = await ctx.get("query", config.DEFAULT_QUERY)
        context_info = {
            "contract_type": contract_type,
            "key_elements": key_elements_list_ctx,
            "cited_laws": cited_laws_list_ctx
        }

        logic_findings_data = extract_json_from_response(ev.logic_issues)
        risk_findings_data = extract_json_from_response(ev.risk_issues)
        logic_findings_list = logic_findings_data.get("logic_findings", [])
        risk_findings_list = risk_findings_data.get("risk_findings", [])
        verified_legal_issues_list = ev.verified_legal_issues
        workflow_step_logger.info(f"Data for synthesis: Legal={len(verified_legal_issues_list)}, Logic={len(logic_findings_list)}, Risk={len(risk_findings_list)}")

        report_sections = {"legal": "", "logic": "", "risk": ""}
        annotations: List[Dict[str, Any]] = []

        legal_details_lines = ["### 2.1. Phân tích Pháp lý (Đối chiếu với Luật)", ""]
        if not verified_legal_issues_list:
            legal_details_lines.append("*Không có điểm pháp lý cụ thể nào được xác định cần kiểm tra thêm.*")
        else:
            for i, issue in enumerate(verified_legal_issues_list):
                verbatim_quote = issue.get('verbatim_quote', 'N/A')
                initial_reason_llm = issue.get('initial_reason', 'N/A')
                initial_reason_llm = re.sub(r"<think>.*?</think>\s*", "", initial_reason_llm, flags=re.DOTALL).strip()
                classified_issue_type = issue.get('classified_issue_type', 'N/A')
                summary = issue.get('verification_summary', 'N/A')
                summary = re.sub(r'\n{2,}', '\n', summary).strip()
                status = issue.get('verification_status', 'UNKNOWN')
                applied_filters = issue.get('rag_filters_applied')

                legal_details_lines.append(f"- **{i+1}. Điểm HĐ cần lưu ý:** `{verbatim_quote}`")
                legal_details_lines.append(f"    - **Gợi ý ban đầu từ LLM:** {initial_reason_llm}")
                legal_details_lines.append(f"    - **Loại nghi vấn được phân loại:** {classified_issue_type}")
                if applied_filters:
                    legal_details_lines.append(f"    - **Filter RAG đã áp dụng:** `{json.dumps(applied_filters, ensure_ascii=False)}`")
                legal_details_lines.append(f"    - **Kết quả Xác minh (RAG - Status: {status}):** {summary}")
                sources_str = ""
                if issue.get('source_nodes'):
                    sources = []
                    for s_node in issue.get('source_nodes', []):
                        score_val = s_node.get('score')
                        score_str = f"{score_val:.2f}" if isinstance(score_val, (int, float)) else "N/A"
                        sources.append(f"{s_node.get('file_name', '?').split('/')[-1]}:Trang {s_node.get('page', '?')} (Score: {score_str})")
                    if sources: sources_str = "; ".join(sources)
                if sources_str:
                    legal_details_lines.append(f"    - *Nguồn tham khảo (RAG):* {sources_str}")
                legal_details_lines.append("---")
                annotations.append({
                    "type": "legal", "status": status, "location_hint": verbatim_quote,
                    "summary": classified_issue_type, "details": summary
                })
        report_sections["legal"] = "\n".join(legal_details_lines)

        logic_details_lines = ["### 2.2. Phân tích Logic & Ngữ nghĩa", ""]
        if not logic_findings_list:
            logic_details_lines.append("*Không phát hiện vấn đề logic hay ngữ nghĩa nào.*")
        else:
            for finding in logic_findings_list:
                finding_type = finding.get('type', 'N/A')
                description = finding.get('description', 'N/A')
                exact_quotes = finding.get('exact_quote', [])
                location_hint = exact_quotes[0] if exact_quotes and isinstance(exact_quotes, list) and exact_quotes[0] else description
                logic_details_lines.append(f"- **Loại:** `{finding_type}`")
                logic_details_lines.append(f"  - **Mô tả:** {description}")
                if exact_quotes and isinstance(exact_quotes, list) and any(q for q in exact_quotes):
                    logic_details_lines.append(f"  - **Trích dẫn liên quan:**")
                    for q_item in exact_quotes:
                        if q_item: logic_details_lines.append(f"    - `{q_item}`")
                elif finding.get('clauses_involved'):
                     logic_details_lines.append(f"  - **Điều khoản liên quan (ước tính):** {json.dumps(finding.get('clauses_involved'), ensure_ascii=False)}")
                elif finding.get('inconsistent_terms'):
                     logic_details_lines.append(f"  - **Thuật ngữ không nhất quán:** {json.dumps(finding.get('inconsistent_terms'), ensure_ascii=False)}")
                elif finding.get('term'):
                     logic_details_lines.append(f"  - **Thuật ngữ không định nghĩa:** {finding.get('term')}")
                annotations.append({
                    "type": "logic", "status": finding_type, "location_hint": location_hint,
                    "summary": description, "details": f"Loại: {finding_type}. Mô tả: {description}"
                })
        report_sections["logic"] = "\n".join(logic_details_lines)

        risk_details_lines = ["### 2.3. Phân tích Rủi ro", ""]
        if not risk_findings_list:
            risk_details_lines.append("*Không phát hiện rủi ro cụ thể nào dựa trên các tiêu chí phân tích.*")
        else:
            for finding in risk_findings_list:
                risk_type = finding.get('risk_type', 'N/A')
                description = finding.get('description', 'N/A')
                suggestion = finding.get('suggestion', 'N/A')
                exact_quote = finding.get('exact_quote')
                clause_involved = finding.get('clause_involved')
                location_hint = exact_quote if exact_quote else (clause_involved if clause_involved else description)
                risk_details_lines.append(f"- **Loại rủi ro:** `{risk_type}`")
                risk_details_lines.append(f"  - **Mô tả:** {description}")
                if exact_quote:
                    risk_details_lines.append(f"  - **Trích dẫn liên quan:** `{exact_quote}`")
                elif clause_involved:
                     risk_details_lines.append(f"  - **Điều khoản liên quan (ước tính):** {clause_involved}")
                risk_details_lines.append(f"  - **Đề xuất:** {suggestion}")
                annotations.append({
                    "type": "risk", "status": risk_type, "location_hint": location_hint,
                    "summary": description, "details": suggestion
                })
        report_sections["risk"] = "\n".join(risk_details_lines)

        num_legal = len(verified_legal_issues_list)
        num_logic = len(logic_findings_list)
        num_risk = len(risk_findings_list)
        legal_highlights = ""
        actionable_legal_suggestions = []
        if num_legal > 0:
            confirmed_issues = [iss for iss in verified_legal_issues_list if "CONFIRMED" in iss.get("verification_status", "")]
            needs_review_issues = [iss for iss in verified_legal_issues_list if "NEEDS_REVIEW" in iss.get("verification_status", "")]
            target_issue_for_highlight = None
            if confirmed_issues: target_issue_for_highlight = confirmed_issues[0]
            elif needs_review_issues: target_issue_for_highlight = needs_review_issues[0]
            else: target_issue_for_highlight = verified_legal_issues_list[0]

            if target_issue_for_highlight:
                legal_highlights = f", ví dụ: {target_issue_for_highlight.get('initial_reason', '')[:50]}... ({target_issue_for_highlight.get('verification_status', '')})"
                if "=>" in target_issue_for_highlight.get('verification_summary', ''):
                    actionable_legal_suggestions.append(target_issue_for_highlight.get('verification_summary').split("=>")[-1].strip())

        logic_highlights = f", ví dụ: {logic_findings_list[0].get('description', '')[:50]}..." if num_logic > 0 and logic_findings_list[0].get('description') else ""
        risk_highlights = f", ví dụ: {risk_findings_list[0].get('description', '')[:50]}..." if num_risk > 0 and risk_findings_list[0].get('description') else ""

        all_suggestions = [finding.get('suggestion') for finding in risk_findings_list if finding.get('suggestion')]
        all_suggestions.extend(actionable_legal_suggestions)
        all_suggestions = [sug for sug in all_suggestions if sug]

        synthesis_prompt = self.prompt_store.get_synthesis_prompt(
            context_info=context_info, query=query, num_legal=num_legal, num_logic=num_logic, num_risk=num_risk,
            legal_highlights=legal_highlights, logic_highlights=logic_highlights, risk_highlights=risk_highlights,
            suggestions=all_suggestions
        )
        workflow_step_logger.info("Calling LLM for Synthesis (Summary & Recommendations Only)...")
        try:
            synthesis_output = await Settings.llm.acomplete(synthesis_prompt)
            synthesis_text = re.sub(r"<think>.*?</think>\s*", "", synthesis_output.text, flags=re.DOTALL).strip()
            workflow_step_logger.info("Synthesis LLM call completed.")
            logger.debug(f"Synthesis LLM raw response:\n{synthesis_text}")
        except Exception as e:
            logger.error(f"Error calling LLM for synthesis: {e}", exc_info=True)
            synthesis_text = "**I. Tóm tắt:**\n(Lỗi: Không thể tạo tóm tắt tự động.)\n\n**V. Tổng hợp Đề xuất:**\n(Lỗi: Không thể tạo đề xuất tự động.)"

        summary_match = re.search(r"\*\*I\. Tóm tắt:\*\*(.*?)(?=\n\*\*V\.|\Z)", synthesis_text, re.DOTALL)
        recs_match = re.search(r"\*\*V\. Tổng hợp Đề xuất:\*\*(.*)", synthesis_text, re.DOTALL)
        final_summary = summary_match.group(1).strip() if summary_match else "(Lỗi: Không trích xuất được Tóm tắt)"
        final_recs = recs_match.group(1).strip() if recs_match else "(Lỗi: Không trích xuất được Đề xuất)"

        final_report_parts = [
            "# Báo cáo Phân tích Hợp đồng", "---", "## 1. Tóm tắt", final_summary, "---",
            "## 2. Phân tích Chi tiết", report_sections["legal"], report_sections["logic"], report_sections["risk"], "---",
            "## 3. Tổng hợp Đề xuất", final_recs, "---",
            "*Lưu ý: Báo cáo này được tạo tự động và chỉ mang tính tham khảo. Cần có sự xem xét của chuyên gia pháp lý.*"
        ]
        final_report = "\n\n".join(final_report_parts)
        workflow_step_logger.info("Final report synthesized.")
        logger.debug(f"Final Report (excerpt):\n{final_report[:500]}...")
        workflow_step_logger.info(f"Extracted {len(annotations)} annotations for UI.")
        logger.debug(f"Annotations sample: {annotations[:2]}")
        self._send_progress(ctx, "Synthesis completed.")
        return StopEvent(result=FinalOutputEvent(report=final_report, annotations=annotations))


async def initialize_analysis_system() -> Optional[Tuple[VectorStoreIndex, BaseTool]]:
    logger.info("Initializing LlamaIndex Settings and Qdrant connection...")
    try:
        Settings.llm = Ollama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, request_timeout=config.REQUEST_TIMEOUT)
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
        logger.info(f"LLM ({config.LLM_MODEL}) and Embeddings ({config.EMBED_MODEL}) configured.")
        try:
            async_qdrant_client = AsyncQdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT, timeout=10)
            await async_qdrant_client.get_collections()
            logger.info("Successfully connected to Qdrant (async check).")
        except Exception as q_err:
            logger.error(f"Failed to connect to Qdrant: {q_err}", exc_info=True)
            raise ConnectionError(f"Could not connect to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}.") from q_err

        vector_store = QdrantVectorStore(aclient=async_qdrant_client, collection_name=config.QDRANT_COLLECTION)
        logger.info(f"QdrantVectorStore initialized for collection '{config.QDRANT_COLLECTION}'.")

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logger.info("VectorStoreIndex created from Qdrant store.")

        query_engine = index.as_query_engine(similarity_top_k=config.SIMILARITY_TOP_K)
        logger.info(f"Query engine created with similarity_top_k={config.SIMILARITY_TOP_K}.")

        law_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=config.RAG_TOOL_NAME,
            description=config.RAG_TOOL_DESC
        )
        logger.info(f"RAG QueryEngineTool '{config.RAG_TOOL_NAME}' created.")
        logger.info("Initialization complete.")
        return index, law_tool # Trả về cả index và law_tool
    except ConnectionError as ce: # Bắt ConnectionError cụ thể từ Qdrant
        logger.error(f"Qdrant connection failed during initialization: {ce}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"Failed to initialize analysis system: {e}", exc_info=True)
        return None
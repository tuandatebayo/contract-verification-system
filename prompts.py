# prompts.py
from typing import List, Dict, Optional, Any

class PromptStore:
    """
    A central store for all LLM prompts used in the contract analysis workflow.
    """

    # --- Constants ---
    NO_LEGAL_ISSUES_FOUND_MSG = "Không xác định được điểm cụ thể nào trong văn bản cần đánh giá pháp lý thêm."

    # --- Prompt Methods ---

    def get_context_analysis_prompt(self, query: str, contract_text: str) -> str:
        """Generates the prompt for initial context analysis."""
        return f"""
        **Yêu cầu:** Dựa **chặt chẽ và duy nhất** vào nội dung hợp đồng dưới đây và yêu cầu kiểm tra '{query}', hãy thực hiện các việc sau:
        1.  **Xác định loại hợp đồng:** Trích xuất **chính xác** tên loại hợp đồng **được nêu rõ** trong văn bản. Nếu không nêu rõ, hãy ghi "Không xác định rõ trong văn bản".
        2.  **Trích xuất yếu tố/điều khoản chính:** Liệt kê các yếu tố hoặc điều khoản **được đề cập trực tiếp** trong văn bản hợp đồng (ví dụ: các bên tham gia, đối tượng hợp đồng, giá trị/tiền lương, thời hạn, quyền và nghĩa vụ cơ bản, điều khoản thanh toán, điều khoản chấm dứt, các điều luật được trích dẫn). Chỉ liệt kê những gì có trong văn bản.
        3.  **Xác định văn bản luật được trích dẫn:** Liệt kê **tên các văn bản luật hoặc nghị định** (ví dụ: Bộ luật Lao động 2019, Luật BHXH 2014, Nghị định 38/2022/NĐ-CP) được **trích dẫn trực tiếp** trong nội dung hợp đồng.

        **Nghiêm cấm:** Không được đưa ra bất kỳ giả định, suy diễn hoặc thông tin nào không được nêu rõ trong văn bản hợp đồng dưới đây.

        **Nội dung hợp đồng:**
        ---
        {contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau, không thêm bất kỳ giải thích nào khác. Nếu một trường thông tin không thể xác định từ văn bản, hãy để giá trị là `null` hoặc danh sách rỗng `[]`.
        {{
          "contract_type": "...",
          "key_elements": ["...", "..."],
          "cited_legal_documents": ["...", "..."]
        }}
        """

    def get_logic_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
        """Generates the prompt for logic and semantic analysis."""
        return f"""
        **Yêu cầu:** Phân tích **chỉ dựa trên văn bản hợp đồng** sau đây để xác định các vấn đề về logic và ngữ nghĩa **có bằng chứng trực tiếp** trong văn bản:
        1.  **Mâu thuẫn nội dung:** Tìm các điều khoản **có nội dung trái ngược trực tiếp** với nhau trong văn bản.
        2.  **Ngôn ngữ mơ hồ:** Xác định các từ ngữ, cụm từ hoặc điều khoản **không rõ ràng về mặt ngữ nghĩa**, có thể dẫn đến nhiều cách hiểu khác nhau **ngay trong ngữ cảnh của hợp đồng**.
        3.  **Thiếu nhất quán:** Tìm sự thiếu nhất quán trong việc sử dụng thuật ngữ, định nghĩa hoặc cấu trúc câu **trong chính văn bản hợp đồng**.

        **Nghiêm cấm:** Không đưa ra bất kỳ giả định, suy diễn hoặc đánh giá nào không dựa trên bằng chứng trực tiếp từ văn bản hợp đồng. Chỉ tập trung vào các vấn đề logic và ngữ nghĩa nội tại của văn bản.

        **Thông tin tham khảo (chỉ để hiểu ngữ cảnh, không dùng để suy diễn):**
        - Loại hợp đồng đã xác định: {contract_type}
        - Bối cảnh/Yêu cầu ban đầu: {query}

        **Nội dung hợp đồng cần phân tích:**
        ---
        {contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy vấn đề nào thuộc các loại trên, trả về danh sách `logic_findings` rỗng `[]`.
        {{
          "logic_findings": [
            {{"type": "contradiction/ambiguity/inconsistency", "description": "Mô tả vấn đề dựa trên văn bản", "clauses_involved": ["Trích dẫn chính xác điều khoản/phần văn bản liên quan 1", "Trích dẫn chính xác điều khoản/phần văn bản liên quan 2 (nếu có)"]}},
            ...
          ]
        }}
        """

    def get_risk_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
        """Generates the prompt for risk analysis."""
        return f"""
        **Yêu cầu:** Dựa **chỉ vào nội dung văn bản hợp đồng** dưới đây và loại hợp đồng '{contract_type}', hãy thực hiện các việc sau:
        1.  **Kiểm tra sự hiện diện của điều khoản:** Xác định xem các điều khoản thường gặp sau đây có **xuất hiện trong văn bản hợp đồng hay không**: phạt vi phạm, bảo mật, bất khả kháng, luật áp dụng, giải quyết tranh chấp. Chỉ liệt kê những điều khoản **không tìm thấy** trong văn bản.
        2.  **Xác định điều khoản có thể bất lợi (dựa trên từ ngữ):** Tìm các điều khoản mà **cách diễn đạt trong văn bản** có vẻ tạo ra sự mất cân bằng rõ rệt về quyền lợi/nghĩa vụ giữa các bên. Trích dẫn cụ thể điều khoản đó.
        3.  **Xác định rủi ro từ nội dung hiện có:** Tìm các rủi ro **phát sinh trực tiếp từ cách các điều khoản hiện có được viết** (ví dụ: nghĩa vụ không rõ ràng, thời hạn không cụ thể).

        **Nghiêm cấm:** Không được đưa ra bất kỳ giả định, suy diễn về rủi ro không dựa trên bằng chứng trực tiếp từ văn bản. Không đánh giá điều khoản là "quan trọng" hay "không quan trọng" trừ khi văn bản tự nêu vậy. Không đề xuất khắc phục dựa trên kiến thức bên ngoài, chỉ đề xuất dựa trên việc làm rõ hoặc sửa đổi các điểm đã xác định **ngay trong văn bản**.

        **Thông tin tham khảo (chỉ để hiểu ngữ cảnh, không dùng để suy diễn):**
        - Yêu cầu ban đầu: {query}

        **Nội dung hợp đồng cần phân tích:**
        ---
        {contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy rủi ro nào thuộc các loại trên, trả về danh sách `risk_findings` rỗng `[]`. Đối với đề xuất, chỉ tập trung vào việc làm rõ/sửa đổi dựa trên vấn đề đã xác định từ văn bản.
        {{
          "risk_findings": [
            {{"risk_type": "missing_clause/potentially_unfavorable_clause/risk_from_wording", "description": "Mô tả rủi ro dựa trên văn bản...", "suggestion": "Đề xuất làm rõ/sửa đổi dựa trên vấn đề..."}},
            ...
          ]
        }}
        """

    def get_legal_review_prompt(self, contract_text: str, cited_laws_list: List[str]) -> str:
        """Generates the prompt for initial legal 'red flag' identification."""
        cited_laws_str = ', '.join(cited_laws_list) if cited_laws_list else 'Không có'
        return f"""
        **Yêu cầu:** Xem xét **kỹ lưỡng từng điều khoản** của văn bản hợp đồng dưới đây, đặc biệt chú ý đến các **trích dẫn pháp luật** (ví dụ: Điều X, Luật Y). Dựa **chỉ vào nội dung văn bản hợp đồng và danh sách các luật được trích dẫn ({cited_laws_str})**, hãy xác định các điểm cụ thể mà:

        a)  Nội dung điều khoản hợp đồng **có vẻ không nhất quán, xung đột, hoặc diễn giải sai** ý nghĩa của chính điều luật/nghị định mà nó trích dẫn.
        b)  Việc **trích dẫn pháp luật có vẻ không chính xác hoặc không tồn tại** (ví dụ: số điều/khoản không hợp lý, tên luật không khớp với nội dung điều khoản).
        c)  Điều khoản **thiếu thông tin hoặc quy định chi tiết** mà luật/nghị định được trích dẫn thường yêu cầu.
        d)  Điều khoản sử dụng **từ ngữ hoặc quy định có thể gây tranh cãi pháp lý** khi đối chiếu với chuẩn mực chung, ngay cả khi không có trích dẫn cụ thể.

        **Nhiệm vụ của bạn là xác định các điểm "cờ đỏ" (red flags) liên quan đến pháp lý dựa trên văn bản, bao gồm cả tính chính xác của các trích dẫn, chứ KHÔNG phải đưa ra kết luận pháp lý cuối cùng.**

        **Nghiêm cấm:**
        *   **KHÔNG liệt kê mọi điều khoản có trích dẫn luật một cách tự động.** Chỉ nêu ra nếu có dấu hiệu (a), (b), (c), hoặc (d).
        *   **KHÔNG tự tạo ra các truy vấn về điều luật không liên quan đến nội dung HĐ.**
        *   **KHÔNG đưa ra giả định** hoặc thông tin bên ngoài văn bản.

        **Nội dung hợp đồng cần xem xét:**
        ---
        {contract_text}
        ---

        **Định dạng trả lời:** Liệt kê các điểm cần kiểm tra pháp lý dưới dạng danh sách. Mỗi điểm cần nêu rõ:
        - **Điều khoản/Nội dung HĐ:** (Trích dẫn ngắn gọn phần văn bản có vấn đề, bao gồm cả trích dẫn luật nếu có)
        - **Lý do cần kiểm tra:** (Giải thích ngắn gọn tại sao nó là 'cờ đỏ' - ví dụ: 'Nội dung có vẻ mâu thuẫn với Điều X Luật Y được trích dẫn', **'Trích dẫn Điều X Luật Y có vẻ không chính xác/không tồn tại'**, 'Thiếu chi tiết Z theo Luật A', 'Cách diễn đạt W có thể gây tranh cãi').

        Nếu **hoàn toàn không xác định được điểm nào** cần kiểm tra dựa trên các tiêu chí trên trong văn bản, trả về **chính xác** chuỗi: "{self.NO_LEGAL_ISSUES_FOUND_MSG}"
        """

    def get_synthesis_prompt(self,
                             context_info: Dict[str, Any],
                             query: str,
                             legal_issues: str,
                             logic_issues: str,
                             risk_issues: str,
                             contract_text: str) -> str:
        """Generates the final synthesis prompt to create the report."""

        # --- Pre-format strings for the prompt ---
        key_elements_joiner = "\n            - "
        cited_laws_joiner = ", "
        # Use .get() for safety and map to str before joining
        key_elements_list = context_info.get('key_elements', [])
        cited_laws_list = context_info.get('cited_laws', [])

        key_elements_str = key_elements_joiner.join(map(str, key_elements_list)) if key_elements_list else 'Không xác định được'
        cited_laws_str = cited_laws_joiner.join(map(str, cited_laws_list)) if cited_laws_list else 'Không có'
        # Truncate contract text safely
        truncated_contract_text = (contract_text[:2997] + '...') if len(contract_text) > 3000 else contract_text

        # --- Synthesis Prompt Template ---
        prompt_template = f"""
        **Yêu cầu:** Tổng hợp thông tin được cung cấp dưới đây để tạo báo cáo phân tích chi tiết dưới dạng Markdown. **Tuyệt đối chỉ sử dụng thông tin đã được cung cấp trong các mục 1, 2, 3.** Không thêm thông tin, đánh giá, hoặc suy diễn mới không có trong đầu vào. Trình bày rõ ràng các phát hiện và kết quả tra cứu (bao gồm cả trường hợp không tìm thấy, lỗi, hoặc không có nguồn). Định dạng Markdown phải sạch sẽ và dễ đọc.

        **1. Thông tin Ngữ cảnh (Đầu vào):**
           - Loại hợp đồng: {context_info.get('contract_type', 'Không xác định')}
           - Yêu cầu kiểm tra ban đầu: {query}
           - Các yếu tố chính đã xác định (Raw): {key_elements_list} # Pass the raw list for context
           - Văn bản luật được trích dẫn trong HĐ: {cited_laws_str} # Use formatted string

        **2. Kết quả Phân tích Chi tiết (Đầu vào):**
           - **Phân tích Pháp lý (Kết quả kiểm tra các điểm đáng ngờ và tra cứu RAG):**
             ```text
             {legal_issues}
             ```
           - **Phân tích Logic & Ngữ nghĩa (Vấn đề nội tại văn bản - JSON):**
             ```json
             {logic_issues}
             ```
           - **Phân tích Rủi ro (Vấn đề tiềm ẩn, điều khoản thiếu/bất lợi - JSON):**
             ```json
             {risk_issues}
             ```

        **3. Nội dung Hợp đồng Gốc (Để tham chiếu nếu cần):**
        ---
        {truncated_contract_text}
        ---

        **Đầu ra mong muốn (Định dạng Markdown):**

        # Báo cáo Phân tích Hợp đồng

        ## 1. Thông tin Chung
        - **Loại hợp đồng:** {context_info.get('contract_type', 'Không xác định')}
        - **Yêu cầu kiểm tra:** {query}
        - **Văn bản luật được trích dẫn:** {cited_laws_str} # Use formatted string
        - **Các yếu tố/điều khoản chính:**
            - {key_elements_str} # Use formatted string with newlines

        ## 2. Phân tích Chi tiết

        ### 2.1. Phân tích Pháp lý (Đối chiếu với Luật)

        *(Trình bày lại từng **Điểm cần lưu ý** từ `legal_issues`. Bao gồm **Lý do kiểm tra**, **Kết quả/Thông tin tra cứu**, **Đánh giá (nếu có)**, và **Nguồn tham khảo**. Nếu `legal_issues` là "{self.NO_LEGAL_ISSUES_FOUND_MSG}", ghi rõ điều đó. Nếu có lỗi, ghi rõ lỗi.)*

        {legal_issues}

        ### 2.2. Phân tích Logic & Ngữ nghĩa (Vấn đề nội tại)

        *(Phân tích JSON trong `logic_issues`. Liệt kê các `logic_findings`. Nếu không có, ghi "Không phát hiện vấn đề logic hay ngữ nghĩa nào từ văn bản." Nếu có lỗi trong JSON, báo lỗi.)*
        *(Ví dụ: **Loại:** contradiction, **Mô tả:** ..., **Liên quan:** ...)*

        ### 2.3. Phân tích Rủi ro

        *(Phân tích JSON trong `risk_issues`. Liệt kê các `risk_findings`. Nếu không có, ghi "Không phát hiện rủi ro cụ thể nào dựa trên các tiêu chí phân tích." Nếu có lỗi trong JSON, báo lỗi.)*
        *(Ví dụ: **Loại rủi ro:** missing_clause, **Mô tả:** ..., **Đề xuất:** ...)*

        ## 3. Tóm tắt & Đề xuất (Nếu có)

        - **Tóm tắt các phát hiện chính:** (Nêu bật các vấn đề quan trọng nhất từ các phân tích trên. Nếu không có phát hiện nào đáng kể, ghi rõ.)
        - **Đề xuất chung:** (Chỉ tổng hợp các đề xuất (`suggestion`) từ `risk_issues`. **Không tự ý thêm đề xuất pháp lý.**)

        ---
        *Lưu ý: Báo cáo này được tạo tự động và chỉ mang tính tham khảo dựa trên thông tin được cung cấp và các quy tắc phân tích. Cần có sự xem xét của chuyên gia pháp lý.*
        """
        # Replace the placeholder message correctly using the class constant
        final_prompt = prompt_template.replace("{no_issues_found_msg}", self.NO_LEGAL_ISSUES_FOUND_MSG)
        return final_prompt
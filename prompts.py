# prompts.py
from typing import List, Dict, Optional, Any
import json

class PromptStore:
    """
    A central store for all LLM prompts used in the contract analysis workflow,
    updated to request verbatim quotes for highlighting.
    """

    # --- Constants ---
    NO_LEGAL_ISSUES_FOUND_MSG = "Không xác định được điểm cụ thể nào trong văn bản cần đánh giá pháp lý thêm."

    # --- Prompt Methods ---

    def get_context_analysis_prompt(self, query: str, contract_text: str) -> str:
        """Generates the prompt for initial context analysis."""
        # (Giữ nguyên prompt này vì không cần trích dẫn nguyên văn ở bước này)
        truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
        return f"""
        **Yêu cầu:** Dựa **chặt chẽ và duy nhất** vào nội dung hợp đồng dưới đây và yêu cầu kiểm tra '{query}', hãy thực hiện các việc sau:
        1.  **Xác định loại hợp đồng:** Trích xuất **chính xác** tên loại hợp đồng **được nêu rõ** trong văn bản (ví dụ: Hợp đồng lao động xác định thời hạn, Hợp đồng dịch vụ,...). Nếu không nêu rõ, hãy ghi "Không xác định rõ trong văn bản".
        2.  **Trích xuất yếu tố/điều khoản chính:** Liệt kê **ngắn gọn** các yếu tố hoặc điều khoản **quan trọng nhất** được đề cập trực tiếp trong văn bản (ví dụ: Bên A, Bên B, Công việc/Dịch vụ, Địa điểm, Thời hạn, Lương/Phí, Bảo hiểm, Chấm dứt HĐ, Các điều luật được trích dẫn). **Chỉ liệt kê tên yếu tố/điều khoản, không cần trích dẫn nội dung chi tiết.**
        3.  **Xác định văn bản luật được trích dẫn:** Liệt kê **chính xác tên và số hiệu** các văn bản luật hoặc nghị định (ví dụ: Bộ luật Lao động 2019, Luật BHXH 2014, Nghị định 38/2022/NĐ-CP) được **trích dẫn trực tiếp** trong nội dung hợp đồng.

        **Nghiêm cấm:** Không được đưa ra bất kỳ giả định, suy diễn hoặc thông tin nào không được nêu rõ trong văn bản hợp đồng dưới đây.

        **Nội dung hợp đồng:**
        ---
        {truncated_contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau, không thêm bất kỳ giải thích nào khác. Nếu một trường thông tin không thể xác định từ văn bản, hãy để giá trị là `null` hoặc danh sách rỗng `[]`.
        ```json
        {{
          "contract_type": "...",
          "key_elements": ["...", "..."],
          "cited_legal_documents": ["...", "..."]
        }}
        ``` /nothink
        """

    def get_logic_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
        """Generates the prompt for logic analysis, requesting exact quotes."""
        truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
        return f"""
        **Yêu cầu:** Phân tích **chỉ dựa trên văn bản hợp đồng** sau đây để xác định các vấn đề logic **có bằng chứng trực tiếp và rõ ràng** trong văn bản:

        1.  **Mâu thuẫn nội dung trực tiếp (Direct Contradiction):** Tìm các cặp điều khoản/câu văn **có nội dung phủ định trực tiếp lẫn nhau**.
        2.  **Thuật ngữ không nhất quán (Inconsistent Terminology):** Xác định trường hợp **cùng một đối tượng/khái niệm** được gọi bằng **hai hoặc nhiều tên khác nhau** mà không có định nghĩa liên kết.
        3.  **Thuật ngữ quan trọng không được định nghĩa (Undefined Key Term):** Xác định các **thuật ngữ cốt lõi** được sử dụng lặp lại nhưng **không có định nghĩa rõ ràng**.

        **Nghiêm cấm:**
        *   Không suy diễn.
        *   Không đánh giá "mơ hồ" chung chung.
        *   Chỉ dựa vào văn bản được cung cấp.

        **Thông tin tham khảo (chỉ để hiểu ngữ cảnh):**
        - Loại hợp đồng: {contract_type}
        - Yêu cầu ban đầu: {query}

        **Nội dung hợp đồng cần phân tích:**
        ---
        {truncated_contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy vấn đề nào, trả về danh sách `logic_findings` rỗng `[]`.
        **QUAN TRỌNG:** Trường `"exact_quote"` **PHẢI** chứa **bản sao chép nguyên văn (verbatim copy-paste)** của (các) câu hoặc đoạn văn bản có vấn đề từ **Nội dung hợp đồng cần phân tích** ở trên.
        ```json
        {{
          "logic_findings": [
            {{
              "type": "direct_contradiction",
              "description": "Mô tả ngắn gọn mâu thuẫn.",
              "exact_quote": [
                  "Trích dẫn nguyên văn chính xác câu/đoạn 1 có mâu thuẫn từ hợp đồng gốc.",
                  "Trích dẫn nguyên văn chính xác câu/đoạn 2 có mâu thuẫn từ hợp đồng gốc."
              ] // List chứa 2 trích dẫn nguyên văn
            }},
            {{
              "type": "inconsistent_term",
              "description": "Mô tả thuật ngữ nào không nhất quán.",
              "inconsistent_terms": ["Thuật ngữ 1", "Thuật ngữ 2"],
              "exact_quote": [
                  "Trích dẫn nguyên văn ví dụ câu/đoạn dùng Thuật ngữ 1.",
                  "Trích dẫn nguyên văn ví dụ câu/đoạn dùng Thuật ngữ 2."
              ] // List chứa các ví dụ trích dẫn nguyên văn
            }},
            {{
               "type": "undefined_key_term",
               "description": "Mô tả thuật ngữ nào không được định nghĩa.",
               "term": "Thuật ngữ không được định nghĩa",
               "exact_quote": [
                   "Trích dẫn nguyên văn một câu/đoạn tiêu biểu sử dụng thuật ngữ này."
               ] // List chứa 1 ví dụ trích dẫn nguyên văn
            }}
          ]
        }}
        ``` /nothink
        """

    def get_risk_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
        """Generates the prompt for risk analysis, requesting exact quotes."""
        truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
        return f"""
        **Yêu cầu:** Dựa **chỉ vào nội dung văn bản hợp đồng** dưới đây và loại hợp đồng '{contract_type}', hãy thực hiện các việc sau với **bằng chứng văn bản rõ ràng**:

        1.  **Kiểm tra điều khoản tiêu chuẩn vắng mặt (Missing Standard Clause):** Xác định các điều khoản tiêu chuẩn (Phạt vi phạm, Bảo mật, Bất khả kháng, Luật áp dụng, Giải quyết tranh chấp) **không tìm thấy tiêu đề tương ứng** trong văn bản.
        2.  **Xác định điều khoản một chiều rõ ràng (Clear One-Sided Clause):** Tìm điều khoản quy định nghĩa vụ/trách nhiệm/quyền lợi **chỉ cho một bên** mà không có đối ứng rõ ràng.
        3.  **Xác định rủi ro từ từ ngữ/thiếu sót cụ thể (Risk from Specific Vague Wording/Omission):** Tìm từ ngữ chủ quan (hợp lý, tốt nhất, theo quy định...) không có tiêu chí cụ thể, hoặc thiếu thông tin cần thiết (đơn vị tiền tệ, ngày tháng...).

        **Nghiêm cấm:**
        *   Không đánh giá "bất lợi" chung chung.
        *   Không suy diễn rủi ro không dựa trên bằng chứng.
        *   Chỉ đề xuất làm rõ/bổ sung dựa trên vấn đề đã xác định.

        **Thông tin tham khảo (chỉ để hiểu ngữ cảnh):**
        - Yêu cầu ban đầu: {query}

        **Nội dung hợp đồng cần phân tích:**
        ---
        {truncated_contract_text}
        ---

        **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy vấn đề nào, trả về danh sách `risk_findings` rỗng `[]`.
        **QUAN TRỌNG:** Trường `"exact_quote"` **PHẢI** chứa **bản sao chép nguyên văn (verbatim copy-paste)** của câu hoặc đoạn văn bản có vấn đề từ **Nội dung hợp đồng cần phân tích** ở trên. Đối với `missing_standard_clause`, `exact_quote` có thể là `null`.
        ```json
        {{
          "risk_findings": [
            {{
              "risk_type": "missing_standard_clause",
              "description": "Không tìm thấy điều khoản tiêu chuẩn: [Tên điều khoản]",
              "exact_quote": null, // Không có quote cho trường hợp này
              "suggestion": "Cân nhắc bổ sung điều khoản về [Tên điều khoản]."
            }},
            {{
              "risk_type": "one_sided_clause",
              "description": "Điều khoản [mô tả ngắn gọn] chỉ quy định cho một bên.",
              "exact_quote": "Trích dẫn nguyên văn chính xác câu/đoạn có vấn đề một chiều.", // String
              "suggestion": "Xem xét lại điều khoản để đảm bảo cân bằng."
            }},
            {{
              "risk_type": "vague_term_risk",
              "description": "Sử dụng thuật ngữ chủ quan '[Từ ngữ mơ hồ]' không có tiêu chí.",
              "exact_quote": "Trích dẫn nguyên văn chính xác câu/đoạn chứa từ ngữ mơ hồ.", // String
              "suggestion": "Cân nhắc định nghĩa rõ ràng hoặc đưa ra tiêu chí cho '[Từ ngữ mơ hồ]'."
            }},
            {{
              "risk_type": "omission_risk",
              "description": "Thiếu thông tin cần thiết: [Mô tả thông tin thiếu].",
              "exact_quote": "Trích dẫn nguyên văn chính xác câu/đoạn bị thiếu thông tin.", // String
              "suggestion": "Cân nhắc bổ sung thông tin về [Thông tin thiếu]."
            }}
          ]
        }}
        ``` /nothink
        """

    def get_legal_review_prompt(self, contract_text: str, cited_laws_list: List[str]) -> str:
        """Generates the prompt for initial legal 'red flag' identification, requesting verbatim quotes."""
        truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
        cited_laws_str = ', '.join(cited_laws_list) if cited_laws_list else 'Không có'
        return f"""
        **Yêu cầu:** Xem xét **kỹ lưỡng từng điều khoản** của văn bản hợp đồng dưới đây, đặc biệt chú ý đến các **trích dẫn pháp luật**. Dựa **chỉ vào nội dung văn bản hợp đồng và danh sách luật được trích dẫn ({cited_laws_str})**, hãy xác định các điểm "cờ đỏ" (red flags) pháp lý theo tiêu chí a, b, c, d dưới đây.

        a)  Nội dung HĐ **có vẻ mâu thuẫn/diễn giải sai** luật được trích dẫn.
        b)  Trích dẫn pháp luật **có vẻ không chính xác/không tồn tại**.
        c)  Điều khoản **thiếu chi tiết** mà luật được trích dẫn yêu cầu.
        d)  Từ ngữ/quy định **có thể gây tranh cãi pháp lý** chung.

        **Nghiêm cấm:**
        *   KHÔNG liệt kê mọi trích dẫn luật. Chỉ nêu nếu có dấu hiệu (a, b, c, d).
        *   KHÔNG tra cứu luật không liên quan.
        *   KHÔNG đưa ra giả định ngoài văn bản.

        **Nội dung hợp đồng cần xem xét:**
        ---
        {truncated_contract_text}
        ---

        **Định dạng trả lời:** Liệt kê các điểm dưới dạng danh sách Markdown. **QUAN TRỌNG:** Mỗi điểm **PHẢI** bắt đầu bằng:
        - **Vị trí/Trích dẫn Nguyên văn:** (Sao chép **NGUYÊN VĂN, chính xác tuyệt đối (verbatim copy-paste)** câu hoặc đoạn văn bản có vấn đề từ hợp đồng gốc.)
        - **Lý do cần kiểm tra:** (Giải thích **ngắn gọn, rõ ràng** tại sao nó là 'cờ đỏ' theo tiêu chí a, b, c, hoặc d - ví dụ: 'Trích dẫn Điều X Luật Y sai cấu trúc/không tồn tại', 'Nội dung trái Điều Z Luật A', 'Thiếu [chi tiết] theo Luật B', 'Từ ngữ "[từ ngữ]" gây tranh cãi').

        Nếu **không xác định được điểm nào**, trả về **chính xác** chuỗi: "{self.NO_LEGAL_ISSUES_FOUND_MSG}" **(Không thêm dòng nào khác).** /nothink
        """

    def get_synthesis_prompt(self,
                             context_info: Dict[str, Any],
                             query: str,
                             num_legal: int,
                             num_logic: int,
                             num_risk: int,
                             legal_highlights: str,
                             logic_highlights: str,
                             risk_highlights: str,
                             suggestions: List[str]
                             ) -> str:
        """Generates the prompt for the final synthesis (Summary & Recommendations)."""
        # (Giữ nguyên prompt tổng hợp này vì nó không trực tiếp yêu cầu trích dẫn)
        suggestions_str = "- " + "\n- ".join(suggestions) if suggestions else "Không có đề xuất cụ thể từ phân tích."
        prompt = f"""
        **Yêu cầu:** Dựa vào thông tin tổng hợp dưới đây, hãy viết **chỉ Phần I (Tóm tắt)** và **Phần V (Tổng hợp Đề xuất)** cho Báo cáo Phân tích Hợp đồng. **KHÔNG được viết lại các phần chi tiết (II, III, IV).** Văn phong chuyên nghiệp, khách quan.

        **Thông tin Ngữ cảnh:**
           - Loại hợp đồng: {context_info.get('contract_type', 'Không xác định')}
           - Yêu cầu kiểm tra ban đầu: {query}
           - Văn bản luật được trích dẫn trong HĐ: {context_info.get('cited_laws', [])}

        **Kết quả Phân tích Tổng hợp:**
           - Phân tích Pháp lý: Tìm thấy {num_legal} điểm cần lưu ý đã qua xác minh RAG{legal_highlights}.
           - Phân tích Logic: Tìm thấy {num_logic} vấn đề{logic_highlights}.
           - Phân tích Rủi ro: Tìm thấy {num_risk} điểm{risk_highlights}.

        **Các đề xuất cụ thể từ Phân tích Rủi ro và Pháp lý (Nếu có hành động rõ ràng):**
        {json.dumps(suggestions, indent=2, ensure_ascii=False)}
        {legal_highlights} # Include legal highlights again for context on actionable items

        **Đầu ra mong muốn:**

        **I. Tóm tắt:**
        (Viết một đoạn tóm tắt **khoảng 3-5 câu**, nêu bật số lượng và loại vấn đề chính được phát hiện từ 3 phần phân tích Pháp lý, Logic, Rủi ro dựa trên thông tin tổng hợp ở trên. Nhấn mạnh các vấn đề nghiêm trọng nếu có, ví dụ: trích dẫn sai luật, thiếu điều khoản quan trọng, mâu thuẫn nội dung.)

        **V. Tổng hợp Đề xuất:**
        (Tổng hợp các đề xuất **khả thi và cụ thể** từ danh sách đề xuất được cung cấp và từ kết quả xác minh pháp lý có yêu cầu hành động rõ ràng (ví dụ: được đánh dấu '=>'). **Chỉ đưa ra đề xuất dựa trên thông tin đã cho.** Phân nhóm đề xuất nếu có thể, ví dụ: Bổ sung điều khoản, Làm rõ điều khoản, Sửa đổi điều khoản.)
        /nothink
        """
        return prompt


# # prompts.py
# from typing import List, Dict, Optional, Any
# import json
# class PromptStore:
#     """
#     A central store for all LLM prompts used in the contract analysis workflow.
#     """

#     # --- Constants ---
#     NO_LEGAL_ISSUES_FOUND_MSG = "Không xác định được điểm cụ thể nào trong văn bản cần đánh giá pháp lý thêm."

#     # --- Prompt Methods ---

#     def get_context_analysis_prompt(self, query: str, contract_text: str) -> str:
#         """Generates the prompt for initial context analysis."""
#         # Giảm độ dài contract_text nếu cần để tránh prompt quá dài
#         truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
#         return f"""
#         **Yêu cầu:** Dựa **chặt chẽ và duy nhất** vào nội dung hợp đồng dưới đây và yêu cầu kiểm tra '{query}', hãy thực hiện các việc sau:
#         1.  **Xác định loại hợp đồng:** Trích xuất **chính xác** tên loại hợp đồng **được nêu rõ** trong văn bản (ví dụ: Hợp đồng lao động xác định thời hạn, Hợp đồng dịch vụ,...). Nếu không nêu rõ, hãy ghi "Không xác định rõ trong văn bản".
#         2.  **Trích xuất yếu tố/điều khoản chính:** Liệt kê **ngắn gọn** các yếu tố hoặc điều khoản **quan trọng nhất** được đề cập trực tiếp trong văn bản (ví dụ: Bên A, Bên B, Công việc/Dịch vụ, Địa điểm, Thời hạn, Lương/Phí, Bảo hiểm, Chấm dứt HĐ, Các điều luật được trích dẫn). **Chỉ liệt kê tên yếu tố/điều khoản, không cần trích dẫn nội dung chi tiết.**
#         3.  **Xác định văn bản luật được trích dẫn:** Liệt kê **chính xác tên và số hiệu** các văn bản luật hoặc nghị định (ví dụ: Bộ luật Lao động 2019, Luật BHXH 2014, Nghị định 38/2022/NĐ-CP) được **trích dẫn trực tiếp** trong nội dung hợp đồng.

#         **Nghiêm cấm:** Không được đưa ra bất kỳ giả định, suy diễn hoặc thông tin nào không được nêu rõ trong văn bản hợp đồng dưới đây.

#         **Nội dung hợp đồng:**
#         ---
#         {truncated_contract_text}
#         ---

#         **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau, không thêm bất kỳ giải thích nào khác. Nếu một trường thông tin không thể xác định từ văn bản, hãy để giá trị là `null` hoặc danh sách rỗng `[]`.
#         ```json
#         {{
#           "contract_type": "...",
#           "key_elements": ["...", "..."],
#           "cited_legal_documents": ["...", "..."]
#         }}
#         ```
#         """

#     def get_logic_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
#         """Generates the prompt for logic and semantic analysis."""
#         truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
#         return f"""
#         **Yêu cầu:** Phân tích **chỉ dựa trên văn bản hợp đồng** sau đây để xác định các vấn đề logic **có bằng chứng trực tiếp và rõ ràng** trong văn bản:

#         1.  **Mâu thuẫn nội dung trực tiếp (Direct Contradiction):** Tìm các cặp điều khoản/câu văn **có nội dung phủ định trực tiếp lẫn nhau** về cùng một sự kiện, con số, thời gian, hoặc nghĩa vụ cụ thể (ví dụ: Điều A nói X, Điều B nói không phải X; Điều C quy định thời hạn 10 ngày, Điều D quy định 30 ngày cho cùng một việc).
#         2.  **Thuật ngữ không nhất quán (Inconsistent Terminology):** Xác định trường hợp **cùng một đối tượng, khái niệm hoặc bên tham gia** (ví dụ: cùng một công ty, sản phẩm, dịch vụ) được gọi bằng **hai hoặc nhiều tên/cụm từ khác nhau** trong các phần khác nhau của hợp đồng mà **không có điều khoản định nghĩa nào liên kết chúng** là một.
#         3.  **Thuật ngữ quan trọng không được định nghĩa (Undefined Key Term):** Xác định các **thuật ngữ cốt lõi** của hợp đồng (ví dụ: Tên các Bên, Đối tượng Hợp đồng chính, Sản phẩm/Dịch vụ cụ thể, Giá trị Hợp đồng, Ngày Hiệu lực) được sử dụng lặp lại nhưng **hoàn toàn không có phần/điều khoản định nghĩa rõ ràng** chúng là gì.

#         **Nghiêm cấm:**
#         *   Không suy diễn về các mâu thuẫn tiềm ẩn hoặc không rõ ràng.
#         *   Không đánh giá sự "mơ hồ" chung chung hoặc các vấn đề về cấu trúc câu phức tạp.
#         *   Không đưa ra bất kỳ giả định nào không dựa trên bằng chứng trực tiếp từ văn bản.

#         **Thông tin tham khảo (chỉ để hiểu ngữ cảnh, không dùng để suy diễn):**
#         - Loại hợp đồng đã xác định: {contract_type}
#         - Bối cảnh/Yêu cầu ban đầu: {query}

#         **Nội dung hợp đồng cần phân tích:**
#         ---
#         {truncated_contract_text}
#         ---

#         **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy vấn đề nào thuộc các loại trên, trả về danh sách `logic_findings` rỗng `[]`.
#         ```json
#         {{
#           "logic_findings": [
#             {{
#               "type": "direct_contradiction",
#               "description": "Mô tả mâu thuẫn trực tiếp dựa trên văn bản.",
#               "clauses_involved": ["Trích dẫn chính xác điều khoản/phần văn bản 1", "Trích dẫn chính xác điều khoản/phần văn bản 2"]
#             }},
#             {{
#               "type": "inconsistent_term",
#               "description": "Mô tả việc sử dụng thuật ngữ không nhất quán.",
#               "inconsistent_terms": ["Thuật ngữ 1", "Thuật ngữ 2"],
#               "clauses_involved": ["Trích dẫn ví dụ điều khoản dùng thuật ngữ 1", "Trích dẫn ví dụ điều khoản dùng thuật ngữ 2"]
#             }},
#             {{
#                "type": "undefined_key_term",
#                "description": "Mô tả thuật ngữ quan trọng không được định nghĩa.",
#                "term": "Thuật ngữ không được định nghĩa",
#                "clauses_involved": ["Trích dẫn ví dụ điều khoản sử dụng thuật ngữ này"]
#             }}
#           ]
#         }}
#         ```
#         """

#     def get_risk_analysis_prompt(self, contract_type: Optional[str], query: str, contract_text: str) -> str:
#         """Generates the prompt for risk analysis."""
#         truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
#         return f"""
#         **Yêu cầu:** Dựa **chỉ vào nội dung văn bản hợp đồng** dưới đây và loại hợp đồng '{contract_type}', hãy thực hiện các việc sau với **bằng chứng văn bản rõ ràng**:

#         1.  **Kiểm tra điều khoản tiêu chuẩn vắng mặt (Missing Standard Clause):** Xác định xem các điều khoản có tiêu đề **chính xác hoặc rất gần đúng** sau đây có **xuất hiện trong văn bản hợp đồng hay không**: Phạt vi phạm (Penalty/Breach), Bảo mật (Confidentiality), Bất khả kháng (Force Majeure), Luật áp dụng (Governing Law), Giải quyết tranh chấp (Dispute Resolution). Chỉ liệt kê những điều khoản **không tìm thấy tiêu đề tương ứng**.
#         2.  **Xác định điều khoản một chiều rõ ràng (Clear One-Sided Clause):** Tìm các điều khoản quy định về nghĩa vụ, trách nhiệm, hình phạt, hoặc quyền lợi (như quyền chấm dứt, quyền thay đổi) mà **chỉ rõ ràng áp đặt hoặc trao cho một bên** mà **không có quy định đối ứng tương tự** cho bên còn lại trong các tình huống tương đương.
#         3.  **Xác định rủi ro từ từ ngữ/thiếu sót cụ thể (Risk from Specific Vague Wording/Omission):**
#             *   Tìm việc sử dụng các **từ ngữ mang tính chủ quan, thiếu định lượng phổ biến** như: "hợp lý", "kịp thời", "tốt nhất", "phù hợp", "đáng kể", "theo quy định" mà **không có định nghĩa hoặc tiêu chí cụ thể đi kèm** trong điều khoản đó hoặc phần định nghĩa.
#             *   Xác định các trường hợp **thiếu thông tin cụ thể cần thiết** một cách rõ ràng, ví dụ: Nêu giá tiền nhưng **không có đơn vị tiền tệ**, quy định nghĩa vụ có thời hạn nhưng **không có ngày bắt đầu/kết thúc cụ thể**.

#         **Nghiêm cấm:**
#         *   Không đánh giá mức độ "bất lợi" hay "mất cân bằng" chung chung.
#         *   Không suy diễn về các rủi ro không dựa trên bằng chứng từ ngữ hoặc thiếu sót cụ thể đã nêu.
#         *   Không đề xuất khắc phục phức tạp, chỉ tập trung vào việc chỉ ra vấn đề và đề xuất làm rõ/bổ sung thông tin còn thiếu.

#         **Thông tin tham khảo (chỉ để hiểu ngữ cảnh, không dùng để suy diễn):**
#         - Yêu cầu ban đầu: {query}

#         **Nội dung hợp đồng cần phân tích:**
#         ---
#         {truncated_contract_text}
#         ---

#         **Định dạng trả lời:** Chỉ trả lời bằng định dạng JSON sau. Nếu không tìm thấy vấn đề nào thuộc các loại trên, trả về danh sách `risk_findings` rỗng `[]`.
#         ```json
#         {{
#           "risk_findings": [
#             {{
#               "risk_type": "missing_standard_clause",
#               "description": "Không tìm thấy điều khoản tiêu chuẩn có tiêu đề tương ứng: [Tên điều khoản]",
#               "suggestion": "Cân nhắc bổ sung điều khoản về [Tên điều khoản]."
#             }},
#             {{
#               "risk_type": "one_sided_clause",
#               "description": "Điều khoản này quy định [nghĩa vụ/quyền] chỉ cho [Bên A/Bên B] mà không có đối ứng rõ ràng.",
#               "clause_involved": "Trích dẫn chính xác điều khoản/phần văn bản liên quan",
#               "suggestion": "Xem xét lại điều khoản để đảm bảo tính cân bằng/có đi có lại nếu phù hợp."
#             }},
#             {{
#               "risk_type": "vague_term_risk",
#               "description": "Sử dụng thuật ngữ chủ quan '[Từ ngữ mơ hồ]' mà không có định nghĩa/tiêu chí cụ thể.",
#               "clause_involved": "Trích dẫn chính xác điều khoản/phần văn bản liên quan",
#               "suggestion": "Cân nhắc định nghĩa rõ ràng hoặc đưa ra tiêu chí cụ thể cho thuật ngữ '[Từ ngữ mơ hồ]'."
#             }},
#             {{
#               "risk_type": "omission_risk",
#               "description": "Thiếu thông tin cụ thể cần thiết: [Mô tả thông tin thiếu (vd: đơn vị tiền tệ, ngày cụ thể)].",
#               "clause_involved": "Trích dẫn chính xác điều khoản/phần văn bản liên quan",
#               "suggestion": "Cân nhắc bổ sung thông tin cụ thể về [Thông tin thiếu] vào điều khoản."
#             }}
#           ]
#         }}
#         ```
#         """

#     def get_legal_review_prompt(self, contract_text: str, cited_laws_list: List[str]) -> str:
#         """Generates the prompt for initial legal 'red flag' identification."""
#         truncated_contract_text = (contract_text[:8000] + '...') if len(contract_text) > 8000 else contract_text
#         cited_laws_str = ', '.join(cited_laws_list) if cited_laws_list else 'Không có'
#         return f"""
#         **Yêu cầu:** Xem xét **kỹ lưỡng từng điều khoản** của văn bản hợp đồng dưới đây, đặc biệt chú ý đến các **trích dẫn pháp luật** (ví dụ: Điều X, Luật Y). Dựa **chỉ vào nội dung văn bản hợp đồng và danh sách các luật được trích dẫn ({cited_laws_str})**, hãy xác định các điểm cụ thể mà:

#         a)  Nội dung điều khoản hợp đồng **có vẻ không nhất quán, xung đột, hoặc diễn giải sai** ý nghĩa của chính điều luật/nghị định mà nó trích dẫn (ví dụ: HĐ nói A, nhưng luật được trích dẫn nói B).
#         b)  Việc **trích dẫn pháp luật có vẻ không chính xác hoặc không tồn tại** (ví dụ: số Điều/Khoản/Điểm không hợp lệ trong cấu trúc luật đó, tên luật không khớp nội dung điều khoản).
#         c)  Điều khoản **thiếu thông tin hoặc quy định chi tiết** mà luật/nghị định được trích dẫn thường yêu cầu một cách rõ ràng (ví dụ: Luật yêu cầu nêu tỷ lệ X, nhưng HĐ chỉ nói chung chung).
#         d)  Điều khoản sử dụng **từ ngữ hoặc quy định có thể gây tranh cãi pháp lý** khi đối chiếu với chuẩn mực chung hoặc quy định pháp luật khác (ngay cả khi không có trích dẫn cụ thể trong điều khoản đó).

#         **Nhiệm vụ của bạn là xác định các điểm "cờ đỏ" (red flags) liên quan đến pháp lý dựa trên văn bản, bao gồm cả tính chính xác của các trích dẫn, chứ KHÔNG phải đưa ra kết luận pháp lý cuối cùng.**

#         **Nghiêm cấm:**
#         *   **KHÔNG liệt kê mọi điều khoản có trích dẫn luật một cách tự động.** Chỉ nêu ra nếu có dấu hiệu (a), (b), (c), hoặc (d).
#         *   **KHÔNG tự tạo ra các truy vấn về điều luật không liên quan đến nội dung HĐ.**
#         *   **KHÔNG đưa ra giả định** hoặc thông tin bên ngoài văn bản.

#         **Nội dung hợp đồng cần xem xét:**
#         ---
#         {truncated_contract_text}
#         ---

#         **Định dạng trả lời:** Liệt kê các điểm cần kiểm tra pháp lý dưới dạng danh sách Markdown. Mỗi điểm cần nêu rõ:
#         - **Điều khoản/Nội dung HĐ:** (Trích dẫn **ngắn gọn nhưng đủ ngữ cảnh** phần văn bản có vấn đề, bao gồm cả trích dẫn luật nếu có)
#         - **Lý do cần kiểm tra:** (Giải thích **ngắn gọn, rõ ràng** tại sao nó là 'cờ đỏ' theo tiêu chí a, b, c, hoặc d - ví dụ: 'Trích dẫn Điều X Luật Y có vẻ không tồn tại/sai cấu trúc', 'Nội dung HĐ có vẻ trái với Điều Z Luật A được trích dẫn', 'Thiếu chi tiết [tên chi tiết] theo yêu cầu của Luật B', 'Cách diễn đạt "[từ ngữ]" có thể gây tranh cãi pháp lý').

#         Nếu **hoàn toàn không xác định được điểm nào** cần kiểm tra dựa trên các tiêu chí trên trong văn bản, trả về **chính xác** chuỗi: "{self.NO_LEGAL_ISSUES_FOUND_MSG}" **(Không thêm bất kỳ dòng nào khác).**
#         """

#     def get_synthesis_prompt(self,
#                              context_info: Dict[str, Any],
#                              query: str,
#                              num_legal: int,
#                              num_logic: int,
#                              num_risk: int,
#                              legal_highlights: str,
#                              logic_highlights: str,
#                              risk_highlights: str,
#                              suggestions: List[str]
#                              ) -> str:
#         """Generates the prompt for the final synthesis (Summary & Recommendations)."""

#         # Format suggestions for the prompt
#         suggestions_str = "- " + "\n- ".join(suggestions) if suggestions else "Không có đề xuất cụ thể từ phân tích rủi ro."

#         prompt = f"""
#         **Yêu cầu:** Dựa vào thông tin tổng hợp dưới đây, hãy viết **chỉ Phần I (Tóm tắt)** và **Phần V (Tổng hợp Đề xuất)** cho Báo cáo Phân tích Hợp đồng. **KHÔNG được viết lại các phần chi tiết (II, III, IV).** Văn phong chuyên nghiệp, khách quan.

#         **Thông tin Ngữ cảnh:**
#            - Loại hợp đồng: {context_info.get('contract_type', 'Không xác định')}
#            - Yêu cầu kiểm tra ban đầu: {query}
#            - Văn bản luật được trích dẫn trong HĐ: {context_info.get('cited_laws', [])}

#         **Kết quả Phân tích Tổng hợp:**
#            - Phân tích Pháp lý: Tìm thấy {num_legal} điểm cần lưu ý đã qua xác minh RAG{legal_highlights}.
#            - Phân tích Logic: Tìm thấy {num_logic} vấn đề{logic_highlights}.
#            - Phân tích Rủi ro: Tìm thấy {num_risk} điểm{risk_highlights}.

#         **Các đề xuất cụ thể từ Phân tích Rủi ro và Pháp lý (Nếu có hành động rõ ràng):**
#         {json.dumps(suggestions, indent=2, ensure_ascii=False)}
#         {legal_highlights} # Include legal highlights again for context on actionable items

#         **Đầu ra mong muốn:**

#         **I. Tóm tắt:**
#         (Viết một đoạn tóm tắt **khoảng 3-5 câu**, nêu bật số lượng và loại vấn đề chính được phát hiện từ 3 phần phân tích Pháp lý, Logic, Rủi ro dựa trên thông tin tổng hợp ở trên. Nhấn mạnh các vấn đề nghiêm trọng nếu có, ví dụ: trích dẫn sai luật, thiếu điều khoản quan trọng, mâu thuẫn nội dung.)

#         **V. Tổng hợp Đề xuất:**
#         (Tổng hợp các đề xuất **khả thi và cụ thể** từ danh sách đề xuất được cung cấp và từ kết quả xác minh pháp lý có yêu cầu hành động rõ ràng (ví dụ: được đánh dấu '=>'). **Chỉ đưa ra đề xuất dựa trên thông tin đã cho.** Phân nhóm đề xuất nếu có thể, ví dụ: Bổ sung điều khoản, Làm rõ điều khoản, Sửa đổi điều khoản.)
#         """
#         return prompt
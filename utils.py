# utils.py
import fitz  # PyMuPDF
from docx import Document
import io
import logging
import re
import json
from typing import List, Dict, Any
import html
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Trích xuất văn bản từ nội dung file PDF (bytes)."""
    text = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                # Thêm xử lý để nối các dòng bị ngắt giữa chừng (tuỳ chọn)
                # page_text = re.sub(r'-\n', '', page_text) # Nối từ bị gạch nối
                text += page_text + "\n"
        logger.info("Successfully extracted text from PDF.")
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
        raise ValueError(f"Could not process PDF file: {e}")
    return text

def extract_text_from_docx(file_content: bytes) -> str:
    """Trích xuất văn bản từ nội dung file DOCX (bytes)."""
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()) # Bỏ qua paragraph trống
        logger.info("Successfully extracted text from DOCX.")
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}", exc_info=True)
        raise ValueError(f"Could not process DOCX file: {e}")
    return text

def extract_text_from_txt(file_content: bytes) -> str:
    """Trích xuất văn bản từ nội dung file TXT (bytes)."""
    try:
        # Thử các encoding phổ biến
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
             try:
                 text = file_content.decode(encoding)
                 logger.info(f"Successfully extracted text from TXT using {encoding}.")
                 return text
             except UnicodeDecodeError:
                 continue
        # Nếu tất cả thất bại, báo lỗi
        logger.error("Could not decode TXT file with common encodings.")
        raise ValueError("Could not decode TXT file with common encodings (utf-8, utf-16, latin-1, cp1252).")
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}", exc_info=True)
        raise ValueError(f"Could not process TXT file: {e}")

def extract_text(file_content: bytes, file_name: str) -> str:
    """Phát hiện loại file và trích xuất văn bản."""
    lower_filename = file_name.lower()
    if lower_filename.endswith(".pdf"):
        return extract_text_from_pdf(file_content)
    elif lower_filename.endswith(".docx"):
        return extract_text_from_docx(file_content)
    elif lower_filename.endswith(".txt"):
        return extract_text_from_txt(file_content)
    else:
        logger.warning(f"Unsupported file type: {file_name}. Only PDF, DOCX, TXT are supported.")
        raise ValueError("Unsupported file type. Please use PDF, DOCX, or TXT.")

def extract_json_from_response(response_text: str) -> dict:
    """
    Attempts to extract a JSON object from a string, handling markdown code blocks.
    Returns an empty dict if extraction fails.
    """
    logger.debug(f"Attempting to extract JSON from: {response_text[:500]}...") # Log phần đầu
    # Tìm kiếm JSON trong khối markdown ```json ... ```
    match = re.search(r"```json\s*({.*?})\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        logger.debug("Found JSON within markdown block.")
    else:
        # Nếu không có khối markdown, thử tìm từ dấu { đầu tiên đến dấu } cuối cùng
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            logger.debug("Found JSON by searching for braces.")
        else:
            # Trường hợp đặc biệt: LLM trả về JSON không có dấu ngoặc nhọn bao ngoài
            try:
                # Thử parse trực tiếp, loại bỏ whitespace thừa
                parsed = json.loads(response_text.strip())
                if isinstance(parsed, dict):
                    logger.debug("Parsed entire response as JSON.")
                    return parsed
                else:
                    logger.warning(f"Parsed entire response but it's not a dict: {type(parsed)}")
                    return {}
            except json.JSONDecodeError:
                logger.warning(f"Could not find valid JSON structure in response: {response_text[:500]}...")
                return {} # Trả về dict rỗng nếu không tìm thấy JSON hợp lệ

    try:
        parsed_json = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON: {str(parsed_json)[:200]}...")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing extracted string: {e}")
        logger.error(f"Problematic JSON string: {json_str}")
        # Cố gắng sửa lỗi phổ biến: dấu phẩy cuối cùng
        json_str_fixed = json_str.strip().rstrip(',')
        try:
            parsed_json_fixed = json.loads(json_str_fixed)
            logger.warning("Successfully parsed JSON after removing trailing comma.")
            return parsed_json_fixed
        except json.JSONDecodeError:
             logger.error("Failed to parse JSON even after fixing trailing comma.")
             return {} # Trả về dict rỗng nếu vẫn không parse được

# --- Helper Function for Highlighting ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def normalize_for_matching(text: str) -> str:
    """
    Chuẩn hóa văn bản để so sánh: loại bỏ dấu câu không cần thiết,
    viết thường, normalize unicode, chuẩn hóa khoảng trắng.
    """
    if not text:
        return ""
    try:
        # NFKD chuẩn hóa dấu tiếng Việt và tách các ký tự tương thích.
        text = unicodedata.normalize("NFKD", text)
        # Loại bỏ các ký tự điều khiển kết hợp (dấu câu sau NFKD) - giữ lại bản chất chữ
        text = "".join(c for c in text if not unicodedata.combining(c))
        # Giữ lại chữ cái, số, khoảng trắng và một số dấu câu cơ bản có thể quan trọng (-)
        # Loại bỏ các dấu câu khác. Thay đổi regex này nếu cần giữ lại nhiều dấu hơn.
        text = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
        # Chuẩn hóa khoảng trắng: thay thế nhiều khoảng trắng bằng một, strip đầu/cuối.
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    except Exception as e:
        # Ghi lại lỗi nhưng cố gắng trả về dạng lowercase để không dừng hoàn toàn
        logger.error(f"Lỗi khi chuẩn hóa văn bản: '{text[:50]}...': {e}")
        return text.lower() # Fallback

def highlight_contract_with_annotations(contract_text: str, annotations: List[Dict[str, Any]]) -> str:
    """
    Highlights contract text based on annotations using HTML spans and tooltips.
    Phiên bản cải tiến với việc tìm kiếm linh hoạt hơn và mapping vị trí tốt hơn.
    """
    if not contract_text:
        return "<p><i>Nội dung hợp đồng trống.</i></p>"
    if not annotations:
        logger.info("Không có annotation nào để highlight.")
        return html.escape(contract_text).replace("\n", "<br>")

    # 1. Escape toàn bộ văn bản gốc một lần để tránh XSS
    escaped_contract_text = html.escape(contract_text)

    # 2. Chuẩn bị Annotations: Lọc, chuẩn hóa hint và sắp xếp
    valid_annotations = []
    for i, anno in enumerate(annotations):
        location_hint_raw = anno.get("location_hint")
        if location_hint_raw:
            normalized_hint = normalize_for_matching(location_hint_raw)
            # Chỉ xử lý nếu hint có nội dung sau chuẩn hóa
            if normalized_hint:
                # Lưu lại để không cần tính toán nhiều lần
                anno['_normalized_hint'] = normalized_hint
                # Dùng độ dài gốc để sắp xếp (ưu tiên match dài hơn trước)
                anno['_raw_hint_len'] = len(location_hint_raw)
                # Thêm id duy nhất để debug nếu cần
                anno['_id'] = i
                valid_annotations.append(anno)
            else:
                 logger.warning(f"Annotation {i} có hint '{location_hint_raw[:30]}...' trở thành rỗng sau chuẩn hóa, bỏ qua.")
        else:
             logger.warning(f"Annotation {i} thiếu 'location_hint', bỏ qua.")


    # Sắp xếp theo độ dài hint gốc giảm dần (quan trọng cho xử lý chồng chéo)
    sorted_annotations = sorted(valid_annotations, key=lambda x: x['_raw_hint_len'], reverse=True)

    # 3. Xác định các đoạn cần highlight và thông tin tag (Pass 1)
    insertions = [] # Lưu thông tin: (escaped_start, escaped_end, start_tag, end_tag, anno_id)
    # Set này theo dõi index trong `escaped_contract_text` đã được "claim" bởi một highlight
    processed_escaped_indices = set()

    for anno in sorted_annotations:
        location_hint_raw = anno["location_hint"]
        location_hint_raw = location_hint_raw.strip('"')
        normalized_hint = anno["_normalized_hint"]
        anno_id = anno["_id"]

        # --- Tìm tất cả các ứng viên trong văn bản gốc và xác minh ---
        best_verified_match = None
        # Regex tìm hint gốc, bỏ qua hoa/thường
        pattern_soft = re.compile(re.escape(location_hint_raw), re.IGNORECASE | re.UNICODE)

        match_candidates = list(pattern_soft.finditer(contract_text))

        if not match_candidates:
            logger.warning(f"[Anno {anno_id} WARN] Hint gốc '{location_hint_raw[:40]}...' không tìm thấy trong văn bản gốc (case-insensitive).")
            continue

        # Tìm ứng viên khớp nhất sau khi chuẩn hóa
        verified_match_found = False
        for match in match_candidates:
            start, end = match.span()
            original_segment = contract_text[start:end]
            normalized_segment = normalize_for_matching(original_segment)

            # Ưu tiên khớp chính xác sau chuẩn hóa
            if normalized_segment == normalized_hint:
                # Tính toán vị trí trong văn bản đã escape
                # Tối ưu: tính độ dài escape một lần cho các phần liên quan
                # Lưu ý: cách tính này giả định html.escape không thay đổi độ dài quá nhiều
                # hoặc các thay đổi được phân bố đều. Cách an toàn nhất là escape từng phần.
                pre_text_escaped_len = len(html.escape(contract_text[:start]))
                match_text_escaped_len = len(html.escape(original_segment)) # Escape segment đã match
                escaped_start = pre_text_escaped_len
                escaped_end = pre_text_escaped_len + match_text_escaped_len

                # Kiểm tra chồng chéo với các highlight đã được chọn trước đó
                is_overlapping = False
                for i in range(escaped_start, escaped_end):
                    if i in processed_escaped_indices:
                        is_overlapping = True
                        break

                if not is_overlapping:
                    # Tìm thấy match tốt, không chồng chéo
                    best_verified_match = match
                    best_escaped_start = escaped_start
                    best_escaped_end = escaped_end
                    verified_match_found = True
                    logger.debug(f"[Anno {anno_id} DEBUG] Xác minh khớp cho '{location_hint_raw[:30]}' tại vị trí gốc {match.span()}, escaped {best_escaped_start}-{best_escaped_end}")
                    break # Lấy match đầu tiên hợp lệ (do đã sort theo độ dài hint)

        # Nếu không tìm được match hợp lệ sau khi xác minh và kiểm tra chồng chéo
        if not verified_match_found:
            if not match_candidates: # Trường hợp không có ứng viên nào từ đầu
                 pass # Đã log warning ở trên
            else: # Có ứng viên nhưng không khớp chuẩn hóa hoặc bị chồng chéo
                 logger.warning(f"[Anno {anno_id} WARN] Không tìm thấy vị trí phù hợp (không chồng chéo và khớp chuẩn hóa) cho hint '{location_hint_raw[:40]}...'.")
            continue # Bỏ qua annotation này

        # --- Lưu thông tin để chèn tag ---
        # Lấy thông tin từ annotation
        summary = html.escape(anno.get('summary', 'N/A'))
        details = html.escape(anno.get('details', 'N/A'))
        status = html.escape(anno.get('status', 'unknown'))
        anno_type = html.escape(anno.get('type', 'general'))

        # Tạo nội dung tooltip và class CSS
        tooltip_text = f"[{anno_type.upper()}/{status}] {summary} || Chi tiết: {details}"
        css_class = f"highlight highlight-{anno_type} status-{status.lower().replace('_', '-')}"

        # Tạo thẻ mở và đóng
        start_tag = f'<span class="{css_class}" title="{tooltip_text}">'
        end_tag = '</span>'

        # Lưu lại thông tin cần thiết
        insertions.append((best_escaped_start, best_escaped_end, start_tag, end_tag, anno_id))

        # Đánh dấu các index trong `escaped_contract_text` đã được sử dụng
        for i in range(best_escaped_start, best_escaped_end):
            processed_escaped_indices.add(i)

        logger.info(f"[Anno {anno_id} INFO] Đã đánh dấu để highlight: '{location_hint_raw[:30]}...' tại vị trí escaped {best_escaped_start}-{best_escaped_end}")

    # 4. Xây dựng chuỗi HTML cuối cùng từ các điểm đã đánh dấu (Pass 2)
    if not insertions: # Nếu không có gì để highlight
        logger.info("Không có annotation nào được áp dụng để highlight.")
        return escaped_contract_text.replace("\n", "<br>")

    # Sắp xếp các điểm chèn theo vị trí bắt đầu để xây dựng tuần tự
    insertions.sort(key=lambda x: x[0])

    final_html_parts = []
    current_pos = 0 # Vị trí đang xử lý trong `escaped_contract_text`

    for esc_start, esc_end, start_tag, end_tag, anno_id in insertions:
        # Thêm phần văn bản gốc (đã escape) từ vị trí cuối cùng đến đầu tag mới
        if esc_start > current_pos:
            final_html_parts.append(escaped_contract_text[current_pos:esc_start])

        # Thêm tag mở
        final_html_parts.append(start_tag)
        # Thêm nội dung gốc (đã escape) bên trong tag
        final_html_parts.append(escaped_contract_text[esc_start:esc_end])
        # Thêm tag đóng
        final_html_parts.append(end_tag)

        # Cập nhật vị trí hiện tại
        current_pos = esc_end

    # Thêm phần văn bản còn lại sau tag cuối cùng
    if current_pos < len(escaped_contract_text):
        final_html_parts.append(escaped_contract_text[current_pos:])

    # Nối các phần lại thành chuỗi HTML hoàn chỉnh
    highlighted_html_final = "".join(final_html_parts)

    # 5. Thay thế ký tự xuống dòng bằng <br> và trả về
    return highlighted_html_final.replace("\n", "<br>")

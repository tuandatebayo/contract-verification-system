# utils.py
import fitz  # PyMuPDF
from docx import Document
import io
import logging
import re
import json
from typing import List, Dict, Any, Optional # Added Optional
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

# --- Helper functions moved from meta_data.py ---
def roman_to_int(s: str) -> Optional[int]:
    if not s or not isinstance(s, str): return None
    s = s.upper()
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000} # Added L, C, D, M
    i = 0; num = 0
    valid_roman = all(char in roman_map for char in s)
    if not valid_roman: return None
    while i < len(s):
        s1 = roman_map.get(s[i], 0)
        if (i + 1) < len(s):
            s2 = roman_map.get(s[i + 1], 0)
            if s1 >= s2: num += s1; i += 1
            else: num += (s2 - s1); i += 2
        else: num += s1; i += 1
    return num if num > 0 else None

def normalize_law_name_simple(text: str) -> Optional[str]:
    """Chuẩn hóa tên luật về dạng đơn giản (từ khóa + năm)."""
    if not text: return None
    text_lower = text.lower()
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text_lower)
    year_str = year_match.group(1) if year_match else ""
    
    common_law_phrases = [
        r"bộ luật", r"luật", r"nghị định", r"thông tư", r"quyết định", r"chỉ thị",
        r"hiến pháp", r"pháp lệnh", r"công văn", r"nghị quyết", r"lệnh", r"thông báo",
        r"số.*?\/.*?\/.*?\b", 
        r"\b(số|no\.?|ngày|dated|về việc|quy định|hướng dẫn|thi hành|thực hiện|ban hành)\b",
        r"của", r"và", r"năm", r"sửa đổi", r"bổ sung", r"thay thế", r"hợp nhất", r"liên tịch",
        r"quốc hội", r"chính phủ", r"bộ tư pháp", r"bộ công an", r"bộ tài chính", 
        r"bộ kế hoạch và đầu tư", r"bộ công thương", 
        r"bộ lao động - thương binh và xã hội", r"bộ lao động thương binh và xã hội",
        r"ngân hàng nhà nước việt nam", r"ngân hàng nhà nước",
        r"bộ trưởng", r"thủ tướng", r"ủy ban thường vụ quốc hội", r"ủy ban thường vụ", 
        r"chủ tịch nước", r"hội đồng thẩm phán tòa án nhân dân tối cao", r"tòa án nhân dân tối cao",
        r"viện kiểm sát nhân dân tối cao",
        r"khóa\s*[ivxlcdm\d]+"
    ]
    
    cleaned_name = text_lower
    for pattern in common_law_phrases:
        cleaned_name = re.sub(pattern, "", cleaned_name, flags=re.IGNORECASE).strip()
    
    cleaned_name = re.sub(r"[\/\-\.\(\),:;\"“”‘’']", " ", cleaned_name)
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()
    
    if cleaned_name.isdigit() and not year_str:
        return None 

    if cleaned_name and year_str:
        result = f"{cleaned_name} {year_str}" if year_str not in cleaned_name else cleaned_name
    elif cleaned_name:
        result = cleaned_name
    elif year_str: 
        result = year_str
    else:
        unaccented_original = ''.join(c for c in text_lower if c.isalnum() or c.isspace())
        unaccented_original = re.sub(r"\s+", " ", unaccented_original).strip()
        return unaccented_original if unaccented_original else None

    return result.strip() if result else None

# --- Helper Function for Highlighting ---
def normalize_for_matching(text: str) -> str:
    """
    Chuẩn hóa văn bản để so sánh: loại bỏ dấu câu không cần thiết,
    viết thường, normalize unicode, chuẩn hóa khoảng trắng.
    """
    if not text:
        return ""
    try:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        text = re.sub(r'[^\w\s\-]', '', text, flags=re.UNICODE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn hóa văn bản: '{text[:50]}...': {e}")
        return text.lower() 

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

    escaped_contract_text = html.escape(contract_text)
    valid_annotations = []
    for i, anno in enumerate(annotations):
        location_hint_raw = anno.get("location_hint")
        if location_hint_raw:
            normalized_hint = normalize_for_matching(location_hint_raw)
            if normalized_hint:
                anno['_normalized_hint'] = normalized_hint
                anno['_raw_hint_len'] = len(location_hint_raw)
                anno['_id'] = i
                valid_annotations.append(anno)
            else:
                logger.warning(f"Annotation {i} có hint '{location_hint_raw[:30]}...' trở thành rỗng sau chuẩn hóa, bỏ qua.")
        else:
            logger.warning(f"Annotation {i} thiếu 'location_hint', bỏ qua.")

    sorted_annotations = sorted(valid_annotations, key=lambda x: x['_raw_hint_len'], reverse=True)
    insertions = [] 
    processed_escaped_indices = set()

    for anno in sorted_annotations:
        location_hint_raw = anno["location_hint"]
        location_hint_raw = location_hint_raw.strip('"') # Strip quotes if present
        normalized_hint = anno["_normalized_hint"]
        anno_id = anno["_id"]
        
        best_verified_match = None
        pattern_soft = re.compile(re.escape(location_hint_raw), re.IGNORECASE | re.UNICODE)
        match_candidates = list(pattern_soft.finditer(contract_text))

        if not match_candidates:
            logger.warning(f"[Anno {anno_id} WARN] Hint gốc '{location_hint_raw[:40]}...' không tìm thấy trong văn bản gốc (case-insensitive).")
            continue

        verified_match_found = False
        for match in match_candidates:
            start, end = match.span()
            original_segment = contract_text[start:end]
            normalized_segment = normalize_for_matching(original_segment)

            if normalized_segment == normalized_hint:
                pre_text_escaped_len = len(html.escape(contract_text[:start]))
                match_text_escaped_len = len(html.escape(original_segment))
                escaped_start = pre_text_escaped_len
                escaped_end = pre_text_escaped_len + match_text_escaped_len

                is_overlapping = False
                for i_idx in range(escaped_start, escaped_end):
                    if i_idx in processed_escaped_indices:
                        is_overlapping = True
                        break
                if not is_overlapping:
                    best_verified_match = match
                    best_escaped_start = escaped_start
                    best_escaped_end = escaped_end
                    verified_match_found = True
                    logger.debug(f"[Anno {anno_id} DEBUG] Xác minh khớp cho '{location_hint_raw[:30]}' tại vị trí gốc {match.span()}, escaped {best_escaped_start}-{best_escaped_end}")
                    break 
        if not verified_match_found:
            if not match_candidates:
                pass 
            else: 
                logger.warning(f"[Anno {anno_id} WARN] Không tìm thấy vị trí phù hợp (không chồng chéo và khớp chuẩn hóa) cho hint '{location_hint_raw[:40]}...'.")
            continue 

        summary = html.escape(anno.get('summary', 'N/A'))
        details = html.escape(anno.get('details', 'N/A'))
        status = html.escape(anno.get('status', 'unknown'))
        anno_type = html.escape(anno.get('type', 'general'))
        tooltip_text = f"[{anno_type.upper()}/{status}] {summary} || Chi tiết: {details}"
        css_class = f"highlight highlight-{anno_type} status-{status.lower().replace('_', '-')}"
        start_tag = f'<span class="{css_class}" title="{tooltip_text}">'
        end_tag = '</span>'
        insertions.append((best_escaped_start, best_escaped_end, start_tag, end_tag, anno_id))
        for i_idx in range(best_escaped_start, best_escaped_end):
            processed_escaped_indices.add(i_idx)
        logger.info(f"[Anno {anno_id} INFO] Đã đánh dấu để highlight: '{location_hint_raw[:30]}...' tại vị trí escaped {best_escaped_start}-{best_escaped_end}")

    if not insertions: 
        logger.info("Không có annotation nào được áp dụng để highlight.")
        return escaped_contract_text.replace("\n", "<br>")

    insertions.sort(key=lambda x: x[0])
    final_html_parts = []
    current_pos = 0 
    for esc_start, esc_end, start_tag, end_tag, anno_id in insertions:
        if esc_start > current_pos:
            final_html_parts.append(escaped_contract_text[current_pos:esc_start])
        final_html_parts.append(start_tag)
        final_html_parts.append(escaped_contract_text[esc_start:esc_end])
        final_html_parts.append(end_tag)
        current_pos = esc_end
    if current_pos < len(escaped_contract_text):
        final_html_parts.append(escaped_contract_text[current_pos:])
    highlighted_html_final = "".join(final_html_parts)
    return highlighted_html_final.replace("\n", "<br>")
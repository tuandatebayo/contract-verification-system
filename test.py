import streamlit as st
from typing import List, Dict, Any
import html
import re
import logging

# Thiết lập logger đơn giản
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function for Highlighting ---
def highlight_contract_with_annotations(contract_text: str, annotations: List[Dict[str, Any]]) -> str:
    """Highlights contract text based on annotations using HTML spans and tooltips."""
    if not contract_text:
        return "<p><i>Nội dung hợp đồng trống.</i></p>"
    if not annotations:
        logger.info("No annotations provided for highlighting.")
        return html.escape(contract_text).replace("\n", "<br>")

    escaped_contract_text = html.escape(contract_text)
    highlighted_html = escaped_contract_text

    valid_annotations = [anno for anno in annotations if anno.get("location_hint")]
    sorted_annotations = sorted(valid_annotations, key=lambda x: len(x.get("location_hint", "")), reverse=True)
    processed_indices = set()

    for anno in sorted_annotations:
        location_hint_raw = anno.get("location_hint", "")
        if not location_hint_raw:
            continue

        location_hint_escaped = html.escape(location_hint_raw)
        summary = html.escape(anno.get('summary', 'N/A'))
        details = html.escape(anno.get('details', 'N/A'))
        status = html.escape(anno.get('status', 'unknown'))
        anno_type = html.escape(anno.get('type', 'general'))

        tooltip_text = f"[{anno_type.upper()}/{status}] {summary} || Chi tiết: {details}"
        css_class = f"highlight highlight-{anno_type} status-{status.lower().replace('_','-')}"

        try:
            pattern = re.compile(re.escape(location_hint_escaped))
            current_pos = 0
            new_html = ""
            matches_found = 0
            last_end = 0

            for match in pattern.finditer(highlighted_html):
                start, end = match.span()
                if any(idx in processed_indices for idx in range(start, end)):
                    continue

                new_html += highlighted_html[last_end:start]
                span_content = highlighted_html[start:end]
                new_html += f'<span class="{css_class}" title="{tooltip_text}">{span_content}</span>'
                for idx in range(start, end):
                    processed_indices.add(idx)
                last_end = end
                matches_found += 1

            new_html += highlighted_html[last_end:]
            highlighted_html = new_html

            if matches_found > 0:
                logger.info(f"Highlighted '{location_hint_raw[:30]}...' ({matches_found} times) with class '{css_class}'")
            else:
                logger.warning(f"Could not find match for hint: '{location_hint_raw[:50]}'")

        except re.error as re_err:
            logger.error(f"Regex error processing hint '{location_hint_escaped}': {re_err}")
        except Exception as high_err:
            logger.error(f"Error during highlighting for hint '{location_hint_escaped}': {high_err}", exc_info=True)

    highlighted_html_with_br = highlighted_html.replace("\n", "<br>")
    return highlighted_html_with_br

# --- Streamlit UI ---
st.set_page_config(page_title="Highlight Hợp Đồng", layout="wide")
st.title("🔍 Kiểm tra Highlight Văn Bản Hợp Đồng")

# --- Nội dung hợp đồng mẫu ---
default_contract_text = """Điều khoản bảo mật phải được tuân thủ nghiêm ngặt. 
Mọi thông tin liên quan đến khách hàng đều được coi là thông tin bảo mật. 
Vi phạm điều khoản bảo mật sẽ dẫn đến xử lý kỷ luật."""

# --- Annotation mẫu ---
sample_annotations = [
    {
        "location_hint": "bảo mật",
        "summary": "Điều khoản bảo mật",
        "details": "Thông tin liên quan đến bảo mật",
        "status": "confirmed",
        "type": "confidentiality"
    },
    {
        "location_hint": "tuân thủ nghiêm ngặt",
        "summary": "Yêu cầu thực hiện",
        "details": "Cần đảm bảo thực hiện theo đúng yêu cầu",
        "status": "pending_review",
        "type": "compliance"
    },
    {
        "location_hint": "xử lý kỷ luật",
        "summary": "Hình thức xử phạt",
        "details": "Áp dụng trong trường hợp vi phạm nghiêm trọng",
        "status": "confirmed",
        "type": "penalty"
    }
]

# --- Giao diện ---
contract_text = st.text_area("📄 Nội dung hợp đồng:", height=250, value=default_contract_text)

if st.button("✨ Highlight"):
    highlighted_html = highlight_contract_with_annotations(contract_text, sample_annotations)

    # Thêm CSS
    st.markdown("""
    <style>
        .highlight {
            padding: 2px 4px;
            border-radius: 5px;
            cursor: pointer;
        }
        .highlight-confidentiality {
            background-color: yellow;
        }
        .highlight-compliance {
            background-color: lightgreen;
        }
        .highlight-penalty {
            background-color: lightcoral;
        }
        .status-confirmed {
            border: 1px solid green;
        }
        .status-pending-review {
            border: 1px dashed orange;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 Kết quả highlight:")
    st.markdown(highlighted_html, unsafe_allow_html=True)

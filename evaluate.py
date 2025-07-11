# evaluate.py
import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
from tqdm import tqdm

# Import các thành phần cốt lõi từ dự án của bạn
import config
from workflow import (
    initialize_analysis_system,
    MultiAgentContractReviewWorkflow,
    WorkflowStartEvent,
    FinalOutputEvent,
)

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Hằng số ---
DATASET_PATH = "/home/tuandatebayo/WorkSpace/pj/Labor Contract Dataset copy.json"
RESULTS_PATH = "evaluation_results.json"


# --- Chức năng Hỗ trợ Đánh giá ---

def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Tải bộ dữ liệu từ file JSON."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Đã tải thành công {len(dataset)} hợp đồng từ {path}")
        return dataset
    except FileNotFoundError:
        logger.error(f"Lỗi: Không tìm thấy file bộ dữ liệu tại '{path}'.")
        return []
    except json.JSONDecodeError:
        logger.error(f"Lỗi: File '{path}' không phải là file JSON hợp lệ.")
        return []

def map_system_category(anno_type: str) -> str:
    """Ánh xạ loại annotation của hệ thống sang category của ground truth."""
    mapping = {
        "legal": "Legal",
        "logic": "Logic",
        "risk": "Risk"
    }
    return mapping.get(anno_type, "Unknown")

def is_match(generated_issue: Dict, ground_truth_issue: Dict) -> bool:
    """
    Kiểm tra xem một issue do hệ thống tạo ra có khớp với ground truth không.
    Đây là một heuristic đơn giản, có thể cải tiến thêm.
    """
    # 1. So khớp category
    system_category = map_system_category(generated_issue.get('type'))
    gt_category = ground_truth_issue.get('category')
    if system_category != gt_category:
        return False

    # 2. So khớp từ khóa trong mô tả
    # (Đây là phần có thể cải tiến bằng các kỹ thuật NLP phức tạp hơn như semantic similarity)
    generated_desc = generated_issue.get('summary', '').lower() + " " + generated_issue.get('details', '').lower()
    gt_desc = ground_truth_issue.get('description', '').lower()

    # Tạo tập hợp các từ (bỏ qua các từ ngắn)
    gen_words = set(w for w in generated_desc.split() if len(w) > 3)
    gt_words = set(w for w in gt_desc.split() if len(w) > 3)

    # Nếu có ít nhất 3 từ chung thì coi là khớp (có thể điều chỉnh ngưỡng này)
    if len(gen_words.intersection(gt_words)) >= 2:
        return True
    
    # Kiểm tra các loại issue cụ thể hơn
    type_mapping = {
        'citation_not_found': ['citation', 'trích dẫn', 'nghị định'],
        'direct_contradiction': ['contradiction', 'mâu thuẫn'],
        'inconsistent_terminology': ['inconsistent', 'không nhất quán', 'thuật ngữ'],
        'undefined_key_term': ['undefined', 'không định nghĩa'],
        'vague_wording': ['vague', 'mơ hồ', 'linh hoạt'],
        'missing_standard_clause': ['missing', 'thiếu', 'vắng mặt'],
        'clear_one_sided_clause': ['one-sided', 'một chiều', 'bất lợi'],
        'missing_mandatory_element': ['mandatory', 'bắt buộc', 'vi phạm'],
    }

    gt_issue_type = ground_truth_issue.get('issue_type')
    if gt_issue_type in type_mapping:
        if any(keyword in generated_desc for keyword in type_mapping[gt_issue_type]):
            return True

    return False

def analyze_results(results: List[Dict[str, Any]]):
    """
    Phân tích kết quả, tính toán các chỉ số và in báo cáo.
    """
    logger.info("\n" + "="*80)
    logger.info("BẮT ĐẦU PHÂN TÍCH KẾT QUẢ ĐÁNH GIÁ")
    logger.info("="*80)

    stats = {
        "Overall": {"TP": 0, "FP": 0, "FN": 0},
        "Legal": {"TP": 0, "FP": 0, "FN": 0},
        "Logic": {"TP": 0, "FP": 0, "FN": 0},
        "Risk": {"TP": 0, "FP": 0, "FN": 0},
    }

    false_positives = []
    false_negatives = []

    for result in results:
        contract_id = result['contract_id']
        generated_issues = result['generated_annotations']
        ground_truth_issues = result['ground_truth_issues']

        # Dùng để theo dõi các issue đã được khớp
        matched_gt_indices = set()
        matched_gen_indices = set()

        # Tìm True Positives (TP)
        for i, gen_issue in enumerate(generated_issues):
            for j, gt_issue in enumerate(ground_truth_issues):
                if j in matched_gt_indices:
                    continue
                if is_match(gen_issue, gt_issue):
                    category = gt_issue['category']
                    stats[category]['TP'] += 1
                    stats['Overall']['TP'] += 1
                    matched_gt_indices.add(j)
                    matched_gen_indices.add(i)
                    break # Chuyển sang issue tiếp theo của hệ thống

        # Tìm False Positives (FP)
        for i, gen_issue in enumerate(generated_issues):
            if i not in matched_gen_indices:
                category = map_system_category(gen_issue.get('type'))
                if category != "Unknown":
                    stats[category]['FP'] += 1
                    stats['Overall']['FP'] += 1
                    false_positives.append({
                        "contract_id": contract_id,
                        "issue": gen_issue
                    })

        # Tìm False Negatives (FN)
        for j, gt_issue in enumerate(ground_truth_issues):
            if j not in matched_gt_indices:
                category = gt_issue['category']
                stats[category]['FN'] += 1
                stats['Overall']['FN'] += 1
                false_negatives.append({
                    "contract_id": contract_id,
                    "issue": gt_issue
                })

    # --- 4.3.2. Experimental Results ---
    print("\n--- 4.3.2. Experimental Results (Kết quả thực nghiệm) ---\n")
    df_stats = pd.DataFrame.from_dict(stats, orient='index')
    print("Bảng thống kê True Positives (TP), False Positives (FP), False Negatives (FN):")
    print(df_stats)

    # --- 4.3.3. Performance Evaluation ---
    print("\n--- 4.3.3. Performance Evaluation (Đánh giá hiệu năng) ---\n")
    performance = {}
    for category, values in stats.items():
        tp = values['TP']
        fp = values['FP']
        fn = values['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        performance[category] = {
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1_score:.2%}"
        }
    
    df_performance = pd.DataFrame.from_dict(performance, orient='index')
    print("Bảng đánh giá hiệu năng (Precision, Recall, F1-Score):")
    print(df_performance)

    # --- 4.3.4. Error Analysis ---
    print("\n--- 4.3.4. Error Analysis (Phân tích lỗi) ---\n")
    
    print(f"Tổng số lỗi False Positives (hệ thống phát hiện sai): {len(false_positives)}")
    if false_positives:
        print("Một vài ví dụ về False Positives:")
        for fp in false_positives[:5]: # Chỉ hiển thị 5 ví dụ đầu
            print(f"  - Contract ID: {fp['contract_id']}")
            print(f"    - Type: {fp['issue'].get('type')}, Summary: {fp['issue'].get('summary')}")
            
    print("-" * 50)
    
    print(f"Tổng số lỗi False Negatives (hệ thống bỏ sót): {len(false_negatives)}")
    if false_negatives:
        print("Một vài ví dụ về False Negatives:")
        for fn in false_negatives[:5]: # Chỉ hiển thị 5 ví dụ đầu
            print(f"  - Contract ID: {fn['contract_id']}")
            print(f"    - Type: {fn['issue'].get('issue_type')}, Description: {fn['issue'].get('description')}")
            
    logger.info("\n" + "="*80)
    logger.info("PHÂN TÍCH KẾT THÚC")
    logger.info("="*80)


async def run_evaluation():
    """
    Hàm chính để chạy toàn bộ quá trình đánh giá.
    """
    # --- 4.3.1. Test Setup and Scenarios ---
    logger.info("--- 4.3.1. Test Setup and Scenarios ---")
    logger.info("Bắt đầu thiết lập môi trường và kịch bản kiểm thử...")
    
    # 1. Tải bộ dữ liệu
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return

    # 2. Khởi tạo hệ thống (LLM, RAG tool, etc.)
    logger.info("Đang khởi tạo hệ thống phân tích (LLM, RAG, ...)")
    analysis_resources = await initialize_analysis_system()
    if not analysis_resources:
        logger.error("Không thể khởi tạo hệ thống phân tích. Dừng quá trình đánh giá.")
        return
    index, law_tool = analysis_resources
    logger.info("Hệ thống đã sẵn sàng.")

    # 3. Kịch bản: lặp qua từng hợp đồng và chạy workflow
    logger.info(f"Kịch bản: Phân tích {len(dataset)} hợp đồng từ bộ dữ liệu.")
    all_results = []
    
    # Chạy workflow cho từng hợp đồng
    start_time = time.time()
    for contract_data in tqdm(dataset, desc="Đang xử lý các hợp đồng"):
        contract_id = contract_data['contract_id']
        contract_text = contract_data['contract_text']
        ground_truth_issues = contract_data['issues']
        
        logger.debug(f"Đang xử lý Contract ID: {contract_id}")

        # Khởi tạo workflow cho mỗi lần chạy
        workflow = MultiAgentContractReviewWorkflow(
            timeout=config.WORKFLOW_TIMEOUT,
            verbose=False # Tắt verbose để log đỡ bị nhiễu
        )

        start_event = WorkflowStartEvent(
            contract_text=contract_text,
            query=config.DEFAULT_QUERY,
            tools=[law_tool] if law_tool else [],
            index=index
        )
        
        try:
            result_output = await workflow.run(**start_event.model_dump())
            
            if isinstance(result_output, FinalOutputEvent):
                all_results.append({
                    "contract_id": contract_id,
                    "generated_report": result_output.report,
                    "generated_annotations": result_output.annotations,
                    "ground_truth_issues": ground_truth_issues
                })
            else:
                logger.warning(f"Contract ID {contract_id} không trả về FinalOutputEvent. Loại kết quả: {type(result_output)}")
                all_results.append({
                    "contract_id": contract_id,
                    "generated_report": "ERROR: Unexpected output type",
                    "generated_annotations": [],
                    "ground_truth_issues": ground_truth_issues
                })
        except Exception as e:
            logger.error(f"Lỗi nghiêm trọng khi xử lý Contract ID {contract_id}: {e}", exc_info=True)
            all_results.append({
                "contract_id": contract_id,
                "generated_report": f"ERROR: Exception during workflow run - {e}",
                "generated_annotations": [],
                "ground_truth_issues": ground_truth_issues
            })

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_contract = total_time / len(dataset) if dataset else 0
    
    logger.info(f"Đã xử lý xong {len(dataset)} hợp đồng trong {total_time:.2f} giây.")
    logger.info(f"Thời gian xử lý trung bình: {avg_time_per_contract:.2f} giây/hợp đồng.")

    # Lưu kết quả thô ra file để có thể phân tích lại mà không cần chạy workflow
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Kết quả đánh giá đã được lưu vào file: {RESULTS_PATH}")
    
    return all_results


if __name__ == "__main__":
    # Chạy quá trình đánh giá
    results_data = asyncio.run(run_evaluation())
    
    # Nếu chạy thành công, tiến hành phân tích
    if results_data:
        analyze_results(results_data)
    else:
        logger.error("Không có kết quả để phân tích.")
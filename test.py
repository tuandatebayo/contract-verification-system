from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import config
import json
def test_rag_filtering(law_normalized: str, article_number: str = None, top_k: int = 5):
    """
    Kiểm tra filtering trong Qdrant với law_normalized_simple và article_number.
    In ra metadata của các kết quả phù hợp.
    """
    # 1. Khởi tạo client & model
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    # 2. Nhập truy vấn mẫu (có thể thay đổi tùy ngữ cảnh)
    query_text = f"Quy định liên quan đến {law_normalized}, điều {article_number or 'bất kỳ'}"
    query_vector = embed_model.get_text_embedding(query_text)

    # 3. Tạo filter
    filter_conditions = [
        FieldCondition(key="law_normalized_simple", match=MatchValue(value=law_normalized))
    ]
    if article_number:
        filter_conditions.append(FieldCondition(key="article_number", match=MatchValue(value=article_number)))

    query_filter = Filter(must=filter_conditions)

    # 4. Gửi truy vấn tới Qdrant
    result = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    # print(result)
    
    # 5. In kết quả
    print(f"🎯 Query: {query_text}")
    print(f"🔍 Filters: {query_filter}")
    print(f"📦 Found {len(result)} results:\n")
    for i, point in enumerate(result, 1):
        payload = point.payload
        print(f"{i}. 📘 {payload.get('law_full_name')} | Điều: {payload.get('article_number')}")
        node_content_raw = point.payload.get('_node_content')
        if node_content_raw:
            node_content = json.loads(node_content_raw)  # Giải mã chuỗi JSON
            text = node_content.get("text")
            print("Văn bản điều luật:")
            print(text)
        print(f"   🔢 Score: {point.score:.4f}\n")

if __name__ == "__main__":
    # Test với các điều kiện khác nhau
    test_rag_filtering("lao động 2019")
    # test_rag_filtering("việc làm 2013", top_k=5)
    # test_rag_filtering("công đoàn 2012", "2", top_k=5)
    # test_rag_filtering("an toàn vệ sinh lao động 2015", top_k=5)
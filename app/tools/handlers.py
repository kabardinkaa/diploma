import json
from pathlib import Path
from typing import Any


KNOWLEDGE_BASE_PATH = Path(__file__).parent / "knowledge_base.json"


def _load_knowledge_base() -> list[dict[str, Any]]:
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def search_knowledge_base(query: str, product: str = "Неизвестно") -> dict[str, Any]:
    """Search local support knowledge base by product, title, keywords and content."""
    knowledge_base = _load_knowledge_base()
    query_lower = query.lower()
    product_lower = product.lower()

    scored_items: list[tuple[int, dict[str, Any]]] = []

    for item in knowledge_base:
        score = 0

        searchable_text = " ".join(
            [
                item.get("product", ""),
                item.get("title", ""),
                " ".join(item.get("keywords", [])),
                item.get("content", ""),
            ]
        ).lower()

        if product != "Неизвестно" and product_lower in item.get("product", "").lower():
            score += 3

        for word in query_lower.split():
            if len(word) >= 3 and word in searchable_text:
                score += 1

        for keyword in item.get("keywords", []):
            if keyword.lower() in query_lower:
                score += 2

        if score > 0:
            scored_items.append((score, item))

    scored_items.sort(key=lambda pair: pair[0], reverse=True)

    results = [
        {
            "id": item["id"],
            "product": item["product"],
            "title": item["title"],
            "content": item["content"],
            "score": score,
        }
        for score, item in scored_items[:3]
    ]

    return {
        "query": query,
        "product": product,
        "found": len(results) > 0,
        "results": results,
    }


TOOL_HANDLERS = {
    "search_knowledge_base": search_knowledge_base,
}

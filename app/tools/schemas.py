from app.prompts.loader import load_tool_description


SEARCH_KNOWLEDGE_BASE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": load_tool_description("search_knowledge_base"),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Краткий поисковый запрос по проблеме сотрудника."
                },
                "product": {
                    "type": "string",
                    "description": "Продукт или система, по которой нужна инструкция.",
                    "enum": [
                        "Корпоративный VPN",
                        "CRM",
                        "Корпоративная почта",
                        "Рабочее место",
                        "Техподдержка",
                        "Неизвестно"
                    ]
                }
            },
            "required": ["query", "product"],
            "additionalProperties": False
        }
    }
}


TOOLS = [SEARCH_KNOWLEDGE_BASE_SCHEMA]

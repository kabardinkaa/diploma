from app.llm.client import ask_support_assistant


TEST_CASES = [
    {
        "name": "Требует tool",
        "query": "Как сотруднику восстановить доступ к VPN?",
    },
    {
        "name": "Не требует tool",
        "query": "Привет, кто ты?",
    },
    {
        "name": "Пограничный случай",
        "query": "У меня что-то не открывается, что делать?",
    },
]


def main() -> None:
    for index, test_case in enumerate(TEST_CASES, start=1):
        print("=" * 80)
        print(f"Тест {index}: {test_case['name']}")
        print(f"Запрос: {test_case['query']}")
        print("-" * 80)

        answer = ask_support_assistant(test_case["query"])

        print(f"Ответ: {answer}")
        print()


if __name__ == "__main__":
    main()

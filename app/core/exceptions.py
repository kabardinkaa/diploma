class LLMError(Exception):
    code = "llm_error"
    message = "Ошибка LLM-сервиса"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message


class LLMRateLimitError(LLMError):
    code = "llm_rate_limit"
    message = "Превышен лимит запросов к LLM-провайдеру"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    message = "LLM-провайдер не ответил за отведённое время"


class LLMAuthError(LLMError):
    code = "llm_auth"
    message = "Ошибка авторизации у LLM-провайдера"
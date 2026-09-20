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


class InfrastructureError(Exception):
    code = "infrastructure_unavailable"
    message = "Обязательная зависимость временно недоступна"

    def __init__(self) -> None:
        super().__init__(self.message)


class RAGInfrastructureError(InfrastructureError):
    code = "rag_unavailable"
    message = "Сервис поиска по базе знаний временно недоступен"


class SafeInputError(Exception):
    code = "invalid_input"
    message = "Некорректные входные данные"
    status_code = 400

    def __init__(self) -> None:
        super().__init__(self.message)


class PayloadTooLargeError(SafeInputError):
    code = "payload_too_large"
    message = "Размер файла превышает допустимый лимит"
    status_code = 413


class UnsupportedFileTypeError(SafeInputError):
    code = "unsupported_file_type"
    message = "Тип файла не поддерживается"
    status_code = 415


class FileConflictError(SafeInputError):
    code = "file_conflict"
    message = "Файл с таким именем уже существует"
    status_code = 409


class IngestionBusyError(SafeInputError):
    code = "ingestion_busy"
    message = "Операция индексации уже выполняется"
    status_code = 409


class InvalidDocumentError(SafeInputError):
    code = "invalid_document"
    message = "Документ повреждён или не соответствует заявленному формату"
    status_code = 422


class UnsafeFilenameError(InvalidDocumentError):
    message = "Недопустимое имя файла"
    status_code = 400


class ReindexLimitError(SafeInputError):
    code = "reindex_limit_exceeded"
    message = "Объём операции переиндексации превышает допустимый лимит"
    status_code = 413

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


TOPICS: dict[str, str] = {
    "access": "Доступы",
    "workplace": "Рабочее место",
    "software": "Корпоративное ПО",
    "errors": "Ошибки и сбои",
    "escalation": "Эскалации",
}


def topics_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()

    for slug, title in TOPICS.items():
        builder.button(
            text=title,
            callback_data=f"topic:{slug}",
        )

    builder.button(
        text="Отмена",
        callback_data="topic:cancel",
    )

    builder.adjust(2, 2, 1, 1)

    return builder.as_markup()
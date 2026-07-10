from uuid import UUID

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def feedback_kb(message_id: UUID) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="👍", callback_data=f"fb:up:{message_id}")
    builder.button(text="👎", callback_data=f"fb:down:{message_id}")
    builder.adjust(2)
    return builder.as_markup()

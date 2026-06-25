import httpx
from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.services.backend_client import BackendClient


router = Router()


def get_owner_external_id(message: Message) -> str:
    if message.from_user is not None:
        return str(message.from_user.id)

    return str(message.chat.id)


@router.message(CommandStart())
async def cmd_start(
    message: Message,
    backend: BackendClient,
) -> None:
    try:
        await backend.get_or_create_chat(
            owner_external_id=get_owner_external_id(message),
            interface="telegram",
        )
    except httpx.HTTPError:
        await message.answer(
            "Не удалось подключиться к backend. Проверь, что chat-сервис запущен."
        )
        return

    await message.answer(
        "Привет! Я Telegram-интерфейс ИИ-ассистента техподдержки.\n\n"
        "Можно просто написать вопрос текстом — я передам его в backend.\n\n"
        "Команды:\n"
        "/help — справка\n"
        "/ask — задать вопрос по теме\n"
        "/clear — очистить историю\n"
        "/cancel — отменить сценарий"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "Доступные команды:\n\n"
        "/start — начать работу\n"
        "/help — справка\n"
        "/ask — выбрать тему и задать вопрос\n"
        "/clear — очистить историю диалога\n"
        "/cancel — отменить активный сценарий\n\n"
        "Обычный текст я отправляю в backend chat-сервис."
    )


@router.message(Command("clear"))
async def cmd_clear(
    message: Message,
    backend: BackendClient,
) -> None:
    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=get_owner_external_id(message),
            interface="telegram",
        )
        await backend.clear_messages(chat_id)
    except httpx.HTTPError:
        await message.answer(
            "Не удалось очистить историю. Проверь, что backend доступен."
        )
        return

    await message.answer("История очищена.")


@router.message(Command("cancel"))
async def cmd_cancel(
    message: Message,
    state: FSMContext,
) -> None:
    current_state = await state.get_state()

    if current_state is None:
        await message.answer("Сейчас нет активного сценария.")
        return

    await state.clear()
    await message.answer("Сценарий отменён.")
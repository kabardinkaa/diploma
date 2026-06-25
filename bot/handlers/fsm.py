import httpx
from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.handlers.commands import get_owner_external_id
from bot.keyboards.inline import TOPICS, topics_kb
from bot.services.backend_client import BackendClient
from bot.states import AskFlow


router = Router()


async def safe_edit_text(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramBadRequest:
        pass


@router.message(Command("ask"))
async def start_ask(
    message: Message,
    state: FSMContext,
) -> None:
    await state.set_state(AskFlow.waiting_for_topic)

    await message.answer(
        "Выбери тему вопроса:",
        reply_markup=topics_kb(),
    )


@router.callback_query(AskFlow.waiting_for_topic, F.data.startswith("topic:"))
async def choose_topic(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    topic_slug = callback.data.removeprefix("topic:") if callback.data else ""

    if topic_slug == "cancel":
        await state.clear()

        if callback.message:
            await callback.message.edit_text("Сценарий отменён.")

        await callback.answer()
        return

    topic_title = TOPICS.get(topic_slug)

    if topic_title is None:
        await callback.answer("Неизвестная тема", show_alert=True)
        return

    await state.update_data(
        topic_slug=topic_slug,
        topic_title=topic_title,
    )
    await state.set_state(AskFlow.waiting_for_question)

    if callback.message:
        await callback.message.edit_text(
            f"Тема: {topic_title}\n\nТеперь напиши вопрос."
        )

    await callback.answer()


@router.message(AskFlow.waiting_for_question, F.text)
async def ask_question(
    message: Message,
    state: FSMContext,
    backend: BackendClient,
) -> None:
    if not message.text:
        await message.answer("Напиши вопрос текстом.")
        return

    data = await state.get_data()
    topic_title = data.get("topic_title", "Без темы")

    prompt = f"Тема: {topic_title}. Вопрос: {message.text}"

    try:
        chat_id = await backend.get_or_create_chat(
            owner_external_id=get_owner_external_id(message),
            interface="telegram",
        )

        answer_message = await message.answer("Отправляю вопрос в backend...")
        buffer = ""

        async for chunk in backend.send_message(chat_id, prompt):
            buffer += chunk

            if buffer.strip():
                await safe_edit_text(answer_message, buffer)

        if not buffer.strip():
            await safe_edit_text(
                answer_message,
                "Backend вернул пустой ответ.",
            )

    except httpx.HTTPError:
        await message.answer(
            "Не удалось получить ответ от backend. Проверь, что chat-сервис запущен."
        )
    except RuntimeError as exc:
        await message.answer(f"Backend вернул ошибку: {exc}")
    finally:
        await state.clear()


@router.message(AskFlow.waiting_for_question)
async def ask_question_not_text(message: Message) -> None:
    await message.answer("Пожалуйста, отправь вопрос текстом или нажми /cancel.")
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass
from html import escape
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.types import InlineKeyboardButton, CallbackQuery
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums.parse_mode import ParseMode
from aiogram.enums.chat_member_status import ChatMemberStatus

# Импорт логики из нашего пакета
from src.classificator import classify_message, train, DATA_FILE

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

bot = Bot(os.getenv("TOKEN"), default=DefaultBotProperties(
    parse_mode=ParseMode.HTML))
dp = Dispatcher()


@dataclass
class SpamRecord:
    original_text: str
    user_id: int
    chat_id: int
    log_msg_id: int
    reason: str


messagesBySession: dict[int, SpamRecord] = {}
message_storage: dict[str, str] = {}


def _blocking_io_save(category: str, text: str):
    """Синхронная запись в файл (будет запущена в потоке)"""
    try:
        # Используем путь к файлу из классификатора
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            clean_text = ' '.join(text.splitlines())
            f.write(f"\n{category} {clean_text}")
    except Exception as e:
        logger.error(f"Ошибка записи датасета: {e}")


def _blocking_train():
    """Синхронное обучение (будет запущено в потоке)"""
    return train()


@dp.message(CommandStart())
async def start(message: types.Message):
    if message.chat.type == 'private':
        await message.reply("Бот предназначен для администрирования групп, а не для ЛС!")


@dp.message(Command("add"))
async def add(message: types.Message, command: CommandObject):
    if str(message.chat.id) != os.getenv("JOURNAL_CHAT_ID"):
        return

    args = command.args
    if not args:
        await message.reply("Пришлите текст для добавления. Пример: /add Текст сообщения")
        return

    msg_uuid = str(uuid.uuid4())
    message_storage[msg_uuid] = args

    kb = InlineKeyboardBuilder()
    kb.add(InlineKeyboardButton(text="🤬 Спам",
           callback_data=f"add:spam:{msg_uuid}"))
    kb.add(InlineKeyboardButton(text="✅ Не спам",
           callback_data=f"add:ham:{msg_uuid}"))

    await message.reply(
        f"<b>Добавление в датасет вручную</b>\n\n<blockquote>{escape(args)}</blockquote>\n\nВыберите категорию:",
        reply_markup=kb.as_markup()
    )


@dp.callback_query(F.data.startswith("add:"))
async def add_callback(callback: CallbackQuery):
    _, category, msg_uuid = callback.data.split(":")
    text = message_storage.get(msg_uuid)

    if not text:
        await callback.answer("❌ Сообщение устарело")
        return

    kb = InlineKeyboardBuilder()
    kb.add(InlineKeyboardButton(text="✅ Подтвердить",
           callback_data=f"confirm_add:{category}:{msg_uuid}"))
    kb.add(InlineKeyboardButton(text="❌ Отмена",
           callback_data=f"cancel_add:{msg_uuid}"))

    await callback.message.edit_text(
        f"Добавить как <b>{category.upper()}</b>?\n\n<blockquote>{escape(text)}</blockquote>",
        reply_markup=kb.as_markup()
    )


@dp.callback_query(F.data.startswith("confirm_add:"))
async def confirm_add(callback: CallbackQuery):
    _, category, msg_uuid = callback.data.split(":")
    text = message_storage.pop(msg_uuid, None)

    if not text:
        await callback.answer("❌ Ошибка данных")
        return

    await callback.message.edit_text("⏳ <b>Сохранение и переобучение модели...</b>")

    await asyncio.to_thread(_blocking_io_save, category, text)
    accuracy = await asyncio.to_thread(_blocking_train)

    await callback.message.edit_text(
        f"✅ Добавлено как <b>{category}</b>.\n"
        f"Модель переобучена.\n"
        f"<b>Точность: {(accuracy * 100):.0f}%</b>\n\n"
        f"<blockquote>{escape(text)}</blockquote>"
    )


@dp.callback_query(F.data.startswith("cancel_add"))
async def cancel_add(callback: CallbackQuery):
    msg_uuid = callback.data.split(":")[-1]
    message_storage.pop(msg_uuid, None)
    await callback.answer("Отменено")
    await callback.message.delete()


@dp.callback_query(F.data.startswith("false:"))
async def false_positive(callback: CallbackQuery):
    msg_id = int(callback.data.split(":")[1])
    record = messagesBySession.get(msg_id)

    if not record:
        await callback.answer("Сообщение не найдено в кэше")
        return

    await callback.message.edit_text("⏳ <b>Обработка ложного срабатывания...</b>")

    await asyncio.to_thread(_blocking_io_save, "ham", record.original_text)
    accuracy = await asyncio.to_thread(_blocking_train)

    await callback.message.edit_text(
        f"✅ <b>Отмечено как ложное.</b>\n"
        f"Датасет обновлен.\n"
        f"<b>Точность: {(accuracy * 100):.0f}%</b>\n\n"
        f"<blockquote>Сообщение: <i>{escape(record.original_text)}</i></blockquote>"
    )
    messagesBySession.pop(msg_id, None)


@dp.callback_query(F.data.startswith("ban:"))
async def ban_user(callback: CallbackQuery):
    msg_id = int(callback.data.split(":")[1])
    record = messagesBySession.get(msg_id)

    if not record:
        await callback.answer("Ошибка данных")
        return

    try:
        await bot.ban_chat_member(record.chat_id, record.user_id)
        await callback.answer("Пользователь заблокирован!")

        await callback.message.edit_text(
            f"{callback.message.html_text}\n\n🔨 <b>Забанен админом {callback.from_user.first_name}</b>",
            reply_markup=None
        )
    except Exception as e:
        await callback.answer(f"Ошибка бана: {e}", show_alert=True)


async def is_admin_or_group(message: types.Message):

    if message.from_user.id == 1087968824:
        return True
    try:
        member = await bot.get_chat_member(message.chat.id, message.from_user.id)
        return member.status in [ChatMemberStatus.ADMINISTRATOR, ChatMemberStatus.CREATOR]
    except:
        return False


@dp.message()
async def check_spam(message: types.Message):
    if message.chat.type == "private" or str(message.chat.id) != os.getenv("CHAT_IDS"):
        return

    if await is_admin_or_group(message):
        return

    text = message.text or message.caption or ""
    log_reason = ""

    is_forward = (
        (message.forward_from and message.forward_from.is_bot) or
        (message.forward_from_chat and message.forward_from_chat.type == "channel")
    )
    my_channel = os.getenv("CHANNEL_ID")
    if is_forward:
        if my_channel and message.forward_from_chat and str(message.forward_from_chat.id) == str(my_channel):
            pass
        else:
            log_reason = "forwarded_spam"

    if not log_reason and text:
        if classify_message(text):
            log_reason = "classify"

    if log_reason:
        await message.delete()
        await send_log(message, log_reason, text)


async def send_log(message: types.Message, reason_code: str, text: str):
    reasons = {
        "classify": "🤖 ML-классификатор (текст)",
        "forwarded_spam": "✉️ Запрещенный репост",
    }
    reason_text = reasons.get(reason_code, reason_code)

    kb = InlineKeyboardBuilder()
    if reason_code == "classify":
        kb.add(InlineKeyboardButton(text="🤔 Ложное",
               callback_data=f"false:{message.message_id}"))
    kb.add(InlineKeyboardButton(text="⛔️ Бан",
           callback_data=f"ban:{message.message_id}"))

    log_text = (
        f"🚫 <b>Удалено сообщение</b>\n"
        f"👤 Юзер: {message.from_user.mention_html()}\n"
        f"📝 Причина: <b>{reason_text}</b>\n\n"
        f"<blockquote>{escape(text)}</blockquote>"
    )

    sent_msg = await bot.send_message(
        os.getenv("JOURNAL_CHAT_ID"),
        log_text,
        reply_markup=kb.as_markup()
    )

    messagesBySession[message.message_id] = SpamRecord(
        original_text=text,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        log_msg_id=sent_msg.message_id,
        reason=reason_text
    )


async def main():
    logger.info("Запуск бота...")
    # При запуске проверяем наличие модели, если нет - учим
    from src.classificator import model, train
    if model is None:
        logger.info("Модель не найдена, запускаем начальное обучение...")
        await asyncio.to_thread(_blocking_train)
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

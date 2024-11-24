import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from HR import SalesGPT, llm

# Использование переменной окружения для токена бота
bot_token = '7833847901:AAF0eit86FNbi8MURxM6JHlYPzruuRQSijw'

sales_agent = None

async def main():
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)

    @dp.message(Command(commands=["start"]))
    async def handle_start(message):
        global sales_agent
        sales_agent = SalesGPT.from_llm(llm, verbose=False)
        sales_agent.seed_agent()  # Инициализация агента
        await message.answer("Бот запущен! Вы можете начать вводить запросы.")

    @dp.message(F.text)
    async def handle_text_message(message):
        if sales_agent is None:
            await message.answer('Используйте команду /start для инициализации.')
            return

        human_message = message.text
        sales_agent.human_step(human_message)  # Добавление сообщения от пользователя
        ai_message = sales_agent.ai_step()  # Генерация ответа от AI
        await message.answer(ai_message)  # Отправка ответа пользователю

    @dp.message(~F.text)
    async def handle_non_text(message):
        await message.answer('Бот принимает только текст.')

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=['message'])

if __name__ == "__main__":
    asyncio.run(main())
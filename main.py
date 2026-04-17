import asyncio
import sys
import os

# Добавляем текущую директорию в путь поиска модулей, 
# чтобы импорты типа 'from src import ...' работали корректно
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.bot import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("\nБот остановлен.")

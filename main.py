import sys
import os

def run_in_current_terminal():
    """
    Запускает RAG.py из папки src в текущем окне терминала.
    Не требует изменений в самом RAG.py.
    """
    # Путь к папке с main.py
    base_dir = os.path.dirname(__file__)
    # Путь к целевому файлу
    script_path = os.path.join(base_dir, 'src', 'RAG.py')

    print("Запуск RAG.py из папки src в текущем терминале...")
    
    # Запускаем файл как скрипт. 
    # Это выполнит весь код уровня модуля внутри RAG.py.
    import runpy
    runpy.run_path(script_path, run_name="__main__")

if __name__ == "__main__":
    run_in_current_terminal()
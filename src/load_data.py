"""
Скрипт для загрузки исходного .xlsb файла и сохранения в Parquet.
"""
import pandas as pd
import os

def main():
    raw_path = 'data/raw/предикторы_ПВХ_2026-08-20 + ретро (обезличенно).xlsb'
    output_path = 'data/processed/data_raw.parquet'

    if not os.path.exists(raw_path):
        print(f"[ERROR] Файл не найден: {raw_path}")
        print("Положите исходный .xlsb файл в папку data/raw/")
        return

    # ВАЖНО: используйте xlsb_reader.py от организаторов!
    # Здесь заглушка, замените на реальную функцию.
    print("Загрузка данных... (замените заглушку на xlsb_reader)")
    
    # df = read_xlsb(raw_path)  <-- раскомментируйте, когда положите reader
    # df.to_parquet(output_path, index=False)
    # print(f"Готово! Сохранено в {output_path}")

if __name__ == "__main__":
    main()

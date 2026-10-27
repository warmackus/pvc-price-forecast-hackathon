import os
import glob
import pandas as pd

def main():
    # Ищем любой .xlsb файл в папке data/raw/
    raw_dir = 'data/raw/'
    files = glob.glob(os.path.join(raw_dir, '*.xlsb'))
    
    if not files:
        print(f"[ERROR] Не найдено ни одного .xlsb файла в папке {raw_dir}")
        print("Пожалуйста, положите файл с данными туда.")
        return
    
    # Берем первый найденный файл (обычно он один)
    raw_path = files[0]
    print(f"Найден файл: {raw_path}")
    print("Читаем данные (это может занять 10-30 секунд)...")

    try:
        # Используем python-calamine для быстрого чтения .xlsb
        df = pd.read_excel(raw_path, engine='calamine')
        
        print(f"✅ Данные загружены! Строк: {len(df)}, Колонок: {len(df.columns)}")
        print("\nПервые 5 строк:")
        print(df.head())
        
        # Проверяем, что колонки называются так, как ожидается в ТЗ
        expected_cols = ['report_date', 'scenario', 'frequency', 'nomenclature_ru', 
                         'prod_version', 'port_name_ru', 'country', 'location', 
                         'basis', 'currency', 'measure_unit', 'price']
        
        # Если колонки на русском или чуть другие — выводим предупреждение
        if not all(col in df.columns for col in expected_cols):
            print("\n⚠️ ВНИМАНИЕ: Названия колонок могут отличаться от ТЗ!")
            print(f"Фактические колонки: {df.columns.tolist()}")
        
        # Сохраняем в быстрый формат Parquet
        os.makedirs('data/processed/', exist_ok=True)
        output_path = 'data/processed/data_raw.parquet'
        df.to_parquet(output_path, index=False)
        print(f"\n✅ Данные сохранены в {output_path}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при чтении файла: {e}")
        print("Попробуйте установить другую библиотеку: pip install pyxlsb")
        print("И заменить engine='calamine' на engine='pyxlsb' в коде.")

if __name__ == "__main__":
    main()
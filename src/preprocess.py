import sys
import os

# Добавляем корень проекта в путь, чтобы работал импорт src.data_access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_access import get_repository

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_and_prepare():
    """
    Загружает сырые данные, фильтрует целевой ряд и драйверы,
    приводит все к недельной сетке.
    """
    repo = get_repository('parquet')
    df = repo.load_raw()

    # 1. Целевой ряд (ПВХ SG-5 Китай)
    target_mask = df['nomenclature_ru'].str.contains('SG-5', case=False, na=False) & \
                  df['nomenclature_ru'].str.contains('Китай', case=False, na=False)
    df_target = df[target_mask].copy()

    if not pd.api.types.is_datetime64_any_dtype(df_target['report_date']):
        df_target['report_date'] = pd.to_datetime(df_target['report_date'], unit='D', origin='1899-12-30')

    df_target_weekly = df_target.groupby(pd.Grouper(key='report_date', freq='W'))['k'].mean().reset_index()
    df_target_weekly.columns = ['date', 'target']

    # 2. Драйверы (Brent, Нафта, Этилен)
    drivers = ['Нефть марки Brent', 'Нафта', 'Этилен']
    df_drivers = df[df['nomenclature_ru'].str.contains('|'.join(drivers), case=False, na=False)].copy()

    if not pd.api.types.is_datetime64_any_dtype(df_drivers['report_date']):
        df_drivers['report_date'] = pd.to_datetime(df_drivers['report_date'], unit='D', origin='1899-12-30')

    driver_weekly_dict = {}
    for driver in drivers:
        mask = df_drivers['nomenclature_ru'].str.contains(driver, case=False, na=False)
        driver_df = df_drivers[mask]
        if len(driver_df) == 0:
            continue
        driver_weekly = driver_df.groupby(pd.Grouper(key='report_date', freq='W'))['k'].mean().reset_index()
        driver_weekly.columns = ['date', driver]
        driver_weekly_dict[driver] = driver_weekly

    # 3. Объединяем все в одну таблицу по дате
    df_wide = df_target_weekly.copy()
    for driver, series in driver_weekly_dict.items():
        df_wide = pd.merge(df_wide, series, on='date', how='left')

    # 4. Заполняем ТОЛЬКО одиночные пропуски (limit=1), чтобы не «замазывать» длинные обрывы
    df_wide = df_wide.ffill(limit=1)

    # 5. Сортируем по дате
    df_wide = df_wide.sort_values('date').reset_index(drop=True)

    return df_wide


def build_features(df_wide):
    """
    Создает признаки на основе широкой таблицы.
    """
    df = df_wide.copy()

    # --- 1. Лаги целевой переменной (1, 2, 4) ---
    for lag in [1, 2, 4]:
        df[f'target_lag_{lag}'] = df['target'].shift(lag)

    # --- 2. Скользящие средние и волатильность (4, 13) ---
    for window in [4, 13]:
        df[f'target_ma_{window}'] = df['target'].shift(1).rolling(window=window).mean()
        df[f'target_std_{window}'] = df['target'].shift(1).rolling(window=window).std()

    # --- 3. Доходности драйверов (логарифмические) ---
    for driver in ['Нефть марки Brent', 'Нафта', 'Этилен']:
        if driver in df.columns:
            df[f'{driver}_return'] = np.log(df[driver] / df[driver].shift(1))

    # --- 4. Лаги драйверов (1, 2, 4) ---
    for driver in ['Нефть марки Brent', 'Нафта', 'Этилен']:
        if driver in df.columns:
            for lag in [1, 2, 4]:
                df[f'{driver}_lag_{lag}'] = df[driver].shift(lag)

    # --- 5. Сезонность ---
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Удаляем строки с NaN
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Загрузка данных...")
    df_wide = load_and_prepare()
    if df_wide is None or len(df_wide) == 0:
        print("[ERROR] Данные не загружены.")
        exit(1)

    print(f"Широкая таблица: {df_wide.shape}")

    print("Создание признаков...")
    df_features = build_features(df_wide)
    print(f"Таблица с признаками: {df_features.shape}")

    # Сохраняем через репозиторий
    repo = get_repository('parquet')
    repo.save_features(df_features)
    print("✅ Сохранено в data/processed/features.parquet")
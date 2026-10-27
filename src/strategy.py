import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_access import get_repository

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Параметры стратегии ---
HOLDOUT_SIZE = 13
THRESHOLD = 0.005
SEASONAL_FILTER = True
WINTER_MONTHS = [12, 1, 2]


def main():
    # 1. Загружаем фичи через репозиторий
    repo = get_repository('parquet')
    df = repo.load_features().sort_values('date').reset_index(drop=True)

    # 2. Берём последние 13 недель
    holdout = df.iloc[-HOLDOUT_SIZE:][['date', 'target']].copy().reset_index(drop=True)

    # 3. Загружаем прогноз
    forecast_path = os.path.join(PROJECT_ROOT, 'submissions/forecast.csv')
    forecast = pd.read_csv(forecast_path)
    forecast['week_start'] = pd.to_datetime(forecast['week_start'])

    # 4. Мержим по дате
    merged = pd.merge(
        holdout,
        forecast[['week_start', 'forecast']],
        left_on='date', right_on='week_start',
        how='inner'
    ).sort_values('date').reset_index(drop=True)

    # 5. Считаем изменение прогноза (моментум)
    merged['forecast_change'] = merged['forecast'].diff()
    merged['month'] = merged['week_start'].dt.month

    # 6. Генерируем позиции
    positions = []
    for i in range(len(merged)):
        if i == 0 or pd.isna(merged['forecast_change'].iloc[i]):
            positions.append(0)
            continue

        current_price = merged['target'].iloc[i]
        if current_price == 0 or pd.isna(current_price):
            positions.append(0)
            continue

        expected_change = merged['forecast_change'].iloc[i] / current_price
        month = merged['month'].iloc[i]

        if SEASONAL_FILTER and month in WINTER_MONTHS and expected_change > THRESHOLD:
            positions.append(0)
        elif expected_change > THRESHOLD:
            positions.append(1)
        elif expected_change < -THRESHOLD:
            positions.append(-1)
        else:
            positions.append(0)

    merged['position'] = positions

    # 7. Сохраняем strategy.csv
    submissions_dir = os.path.join(PROJECT_ROOT, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    strategy_out = merged[['week_start', 'position']].copy()
    strategy_path = os.path.join(submissions_dir, 'strategy.csv')
    strategy_out.to_csv(strategy_path, index=False)
    print(f"✅ Стратегия сохранена в {strategy_path}")
    print(f"Распределение позиций: {strategy_out['position'].value_counts().to_dict()}")

    # 8. P&L
    merged['price_change'] = merged['target'].diff()
    merged['pnl'] = (merged['position'] * merged['price_change']).fillna(0)

    total_pnl = merged['pnl'].sum()
    cum_pnl = merged['pnl'].cumsum()
    max_drawdown = (cum_pnl - cum_pnl.cummax()).min()
    bh_pnl = merged['price_change'].fillna(0).sum()

    print(f"\n📊 Результаты стратегии:")
    print(f"  Суммарный P&L: {total_pnl:.4f}")
    print(f"  Buy & Hold P&L: {bh_pnl:.4f}")
    print(f"  Разница (Strategy - B&H): {total_pnl - bh_pnl:.4f}")
    print(f"  Максимальная просадка: {max_drawdown:.4f}")
    print(f"  Сделок: {(merged['position'] != 0).sum()} из {len(merged)}")

    pnl_path = os.path.join(submissions_dir, 'pnl_report.csv')
    merged[['week_start', 'target', 'forecast', 'forecast_change',
            'position', 'price_change', 'pnl']].to_csv(pnl_path, index=False)
    print(f"✅ P&L отчёт сохранён в {pnl_path}")


if __name__ == "__main__":
    main()
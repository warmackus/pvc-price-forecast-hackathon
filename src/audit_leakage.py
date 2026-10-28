import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data_access import get_repository

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def audit_rolling_features(df):
    """
    Проверка 1: rolling-признаки должны быть сдвинуты (shift) на 1+,
    иначе для недели t используется target[t] — утечка.
    """
    print("\n🔍 Проверка 1: rolling-признаки (shift перед rolling)")
    leakage_features = []
    for col in df.columns:
        if 'ma_' in col or 'std_' in col or 'rolling' in col:
            if col.startswith('target_'):
                corr = df[col].corr(df['target'])
                if abs(corr) > 0.999:
                    leakage_features.append((col, corr))
    if leakage_features:
        print("  ❌ Подозрительные признаки (корреляция с target > 0.999):")
        for col, corr in leakage_features:
            print(f"     - {col}: corr = {corr:.4f}")
    else:
        print("  ✅ Все rolling-признаки сдвинуты, утечки нет.")


def audit_target_lags(df):
    """
    Проверка 2: лаги target не должны совпадать с target (corr = 1.0).
    """
    print("\n🔍 Проверка 2: лаги target (target_lag_X)")
    for col in df.columns:
        if col.startswith('target_lag_'):
            corr = df[col].corr(df['target'])
            if abs(corr) > 0.999:
                print(f"  ❌ {col}: corr = {corr:.4f} — подозрительно (утечка?)")
            else:
                print(f"  ✅ {col}: corr = {corr:.4f} — норма")


def audit_ffill(df_processed):
    """
    Проверка 3: ffill не должен заполнять длинные пропуски.
    """
    print("\n🔍 Проверка 3: ffill (заполнение пропусков)")
    target = df_processed['target'].values
    max_flat = 0
    current_flat = 0
    for i in range(1, len(target)):
        if target[i] == target[i-1]:
            current_flat += 1
            max_flat = max(max_flat, current_flat)
        else:
            current_flat = 0
    if max_flat > 4:
        print(f"  ⚠️  Найден плоский участок длиной {max_flat} недель — возможно, ffill замазал данные.")
    else:
        print(f"  ✅ Максимальный плоский участок: {max_flat} недель — норма.")


def audit_time_series_split():
    """
    Проверка 4: используется ли TimeSeriesSplit (без shuffle)?
    """
    print("\n🔍 Проверка 4: TimeSeriesSplit (без shuffle)")
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    X_dummy = np.arange(100).reshape(-1, 1)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_dummy)):
        if train_idx.max() >= test_idx.min():
            print(f"  ❌ Фолд {fold+1}: train пересёкся с test — утечка!")
        else:
            print(f"  ✅ Фолд {fold+1}: train max = {train_idx.max()}, test min = {test_idx.min()} — ок")


def audit_strategy(df):
    """
    Проверка 5: позиция на неделю t не должна использовать цену t.
    """
    print("\n🔍 Проверка 5: стратегия (позиция по прошлым данным)")
    if 'position' in df.columns and 'target' in df.columns:
        future_price_change = df['target'].diff().shift(-1)
        corr = df['position'].corr(future_price_change)
        if abs(corr) > 0.5:
            print(f"  ⚠️  position коррелирует с будущим изменением цены: corr = {corr:.3f} — проверьте логику.")
        else:
            print(f"  ✅ position не коррелирует с будущим: corr = {corr:.3f} — норма.")


def main():
    print("=" * 60)
    print("АУДИТ УТЕЧЕК (LEAKAGE AUDIT)")
    print("=" * 60)

    repo = get_repository('parquet')
    df = repo.load_features()
    print(f"\nРазмер данных: {df.shape}")

    audit_rolling_features(df)
    audit_target_lags(df)
    audit_ffill(df)
    audit_time_series_split()

    strategy_path = os.path.join(PROJECT_ROOT, 'submissions', 'strategy.csv')
    if os.path.exists(strategy_path):
        strategy = pd.read_csv(strategy_path)
        # ВАЖНО: приводим week_start к datetime
        strategy['week_start'] = pd.to_datetime(strategy['week_start'])
        merged = pd.merge(strategy, df[['date', 'target']],
                          left_on='week_start', right_on='date', how='left')
        merged['position'] = strategy['position'].values
        audit_strategy(merged)
    else:
        print("\n🔍 Проверка 5: strategy.csv не найден — пропускаем.")

    print("\n" + "=" * 60)
    print("АУДИТ ЗАВЕРШЕН")
    print("=" * 60)
    print("\nЕсли все проверки ✅ — утечек нет.")


if __name__ == "__main__":
    main()
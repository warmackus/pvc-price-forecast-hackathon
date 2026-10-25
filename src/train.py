"""
Здесь мы обучаем LightGBM и валидируем через TimeSeriesSplit.
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

def main():
    # 1. Загрузка фичей
    # df = pd.read_parquet('data/processed/features.parquet')
    
    # 2. Разделение на train/test по времени
    # 3. Обучение LightGBM
    # 4. Оценка sMAPE
    # 5. Сохранение прогноза в submissions/forecast.csv
    print("Training pipeline started...")

if __name__ == "__main__":
    main()

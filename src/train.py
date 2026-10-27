import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from src.data_access import get_repository

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def smape(y_true, y_pred):
    """Симметричная средняя абсолютная процентная ошибка (sMAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100


def main():
    # 1. Загрузка фичей через репозиторий
    print("Загрузка признаков...")
    repo = get_repository('parquet')
    df = repo.load_features()
    print(f"Размер: {df.shape}")

    # 2. Определяем признаки (X) и цель (y)
    feature_cols = [col for col in df.columns if col not in ['date', 'target']]
    X = df[feature_cols]
    y = df['target']

    print(f"Признаков: {len(feature_cols)}")

    # 3. Валидация через TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    smape_scores = []
    mape_scores = []
    smape_ensemble_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- 4. Обучение LightGBM ---
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )

        y_pred_lgbm = model.predict(X_test)

        # --- 5. Обучение Random Forest ---
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        # --- 6. Ансамбль ---
        y_pred_ensemble = 0.7 * y_pred_lgbm + 0.3 * y_pred_rf

        # --- 7. Оценка ---
        smape_lgbm = smape(y_test, y_pred_lgbm)
        smape_ensemble = smape(y_test, y_pred_ensemble)
        mape_lgbm = mean_absolute_percentage_error(y_test, y_pred_lgbm) * 100

        smape_scores.append(smape_lgbm)
        mape_scores.append(mape_lgbm)
        smape_ensemble_scores.append(smape_ensemble)

        print(f"Фолд {fold+1}: sMAPE (LGBM) = {smape_lgbm:.2f}%, sMAPE (Ensemble) = {smape_ensemble:.2f}%")

    print(f"\nСредний sMAPE (LGBM) по фолдам: {np.mean(smape_scores):.2f}%")
    print(f"Средний sMAPE (Ensemble) по фолдам: {np.mean(smape_ensemble_scores):.2f}%")
    print(f"Средний MAPE (LGBM) по фолдам: {np.mean(mape_scores):.2f}%")

    # 8. Финальная модель на всех данных (кроме последних 13 недель)
    print("\nОбучение финальной модели...")

    holdout_size = 13
    X_train_full = X.iloc[:-holdout_size]
    y_train_full = y.iloc[:-holdout_size]
    X_holdout = X.iloc[-holdout_size:]
    y_holdout = y.iloc[-holdout_size:]

    final_lgbm = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=15,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    final_lgbm.fit(X_train_full, y_train_full)
    y_pred_lgbm_holdout = final_lgbm.predict(X_holdout)

    final_rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    final_rf.fit(X_train_full, y_train_full)
    y_pred_rf_holdout = final_rf.predict(X_holdout)

    y_pred_holdout = 0.7 * y_pred_lgbm_holdout + 0.3 * y_pred_rf_holdout

    smape_holdout_lgbm = smape(y_holdout, y_pred_lgbm_holdout)
    smape_holdout_ensemble = smape(y_holdout, y_pred_holdout)

    print(f"sMAPE на holdout (LGBM): {smape_holdout_lgbm:.2f}%")
    print(f"sMAPE на holdout (Ensemble): {smape_holdout_ensemble:.2f}%")

    # 9. Сохранение прогноза
    submissions_dir = os.path.join(PROJECT_ROOT, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)

    submission = pd.DataFrame({
        'week_start': df['date'].iloc[-holdout_size:].values,
        'horizon': range(1, holdout_size + 1),
        'forecast': y_pred_holdout
    })

    forecast_path = os.path.join(submissions_dir, 'forecast.csv')
    submission.to_csv(forecast_path, index=False)
    print(f"✅ Прогноз сохранен в {forecast_path}")

    # 10. Важность признаков
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_lgbm.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nТоп-10 важных признаков (LightGBM):")
    print(importance.head(10).to_string(index=False))

    importance_path = os.path.join(submissions_dir, 'feature_importance.csv')
    importance.to_csv(importance_path, index=False)
    print(f"✅ Важность признаков сохранена в {importance_path}")

    # 11. Расчет sMAPE по КАЖДОМУ горизонту (1..13)
    print("\n📊 sMAPE по каждому горизонту (1–13 недель):")
    horizon_smape = []
    for h in range(1, 14):
        y_true_h = y_holdout.iloc[h-1:h]
        y_pred_h = y_pred_holdout[h-1:h]
        smape_h = smape(y_true_h, y_pred_h)
        horizon_smape.append({'horizon': h, 'smape': smape_h})
        print(f"  Горизонт {h:2d} нед.: sMAPE = {smape_h:6.2f}%")

    avg_smape = np.mean([x['smape'] for x in horizon_smape])
    print(f"\n  Средний sMAPE по всем 13 неделям: {avg_smape:.2f}%")

    horizon_df = pd.DataFrame(horizon_smape)
    horizon_path = os.path.join(submissions_dir, 'horizon_smape.csv')
    horizon_df.to_csv(horizon_path, index=False)
    print(f"✅ Ошибка по горизонтам сохранена в {horizon_path}")


if __name__ == "__main__":
    main()
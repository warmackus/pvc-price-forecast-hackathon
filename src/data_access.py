from abc import ABC, abstractmethod
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataRepository(ABC):
    """Абстрактный интерфейс доступа к данным."""

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_features(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_features(self, df: pd.DataFrame) -> None:
        pass


class ParquetRepository(DataRepository):
    """Реализация на Parquet (по умолчанию, для хакатона)."""

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.join(PROJECT_ROOT, 'data/processed')

    def load_raw(self) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.base_path, 'data_raw.parquet'))

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(os.path.join(self.base_path, 'features.parquet'))

    def save_features(self, df: pd.DataFrame) -> None:
        os.makedirs(self.base_path, exist_ok=True)
        df.to_parquet(os.path.join(self.base_path, 'features.parquet'), index=False)


class PostgresRepository(DataRepository):
    """
    Реализация на PostgreSQL (для продакшена).
    Не используется в хакатоне, но демонстрирует расширяемость.
    Требует: pip install sqlalchemy psycopg2-binary
    """

    def __init__(self, connection_string: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(connection_string)

    def load_raw(self) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM raw_prices', self.engine)

    def load_features(self) -> pd.DataFrame:
        return pd.read_sql('SELECT * FROM features', self.engine)

    def save_features(self, df: pd.DataFrame) -> None:
        df.to_sql('features', self.engine, if_exists='replace', index=False)


def get_repository(backend: str = 'parquet', **kwargs) -> DataRepository:
    """
    Фабрика репозиториев.
    backend='parquet' — для хакатона (по умолчанию).
    backend='postgres' — для продакшена.
    """
    if backend == 'parquet':
        return ParquetRepository(**kwargs)
    elif backend == 'postgres':
        return PostgresRepository(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
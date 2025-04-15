import os
import sqlite3
from pathlib import Path
import pandas as pd
import timesfm
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Абсолютный путь к базе
BASE_DIR = Path(__file__).parent.resolve()
DB_PATH = BASE_DIR / "dataset" / "crypto_data.db"
TABLE_NAME = "btc_data"

# Проверка и инициализация базы
def init_db():
    os.makedirs(DB_PATH.parent, exist_ok=True)
    try:
        with sqlite3.connect(f"file:{DB_PATH}?mode=rw", uri=True) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error as e:
        print(f"Ошибка при проверке базы данных: {e}")
        raise

init_db()

# Загрузка данных из таблицы
def load_data():
    try:
        if not os.access(str(DB_PATH), os.R_OK):
            raise HTTPException(status_code=403, detail="Нет прав доступа к файлу базы данных")

        with sqlite3.connect(f"file:{DB_PATH}?mode=rw", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_NAME,)
            )
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"Таблица {TABLE_NAME} не найдена")

            # Берем последние 1000 записей в правильном порядке
            df = pd.read_sql(
                f"""
                SELECT * FROM (
                    SELECT * FROM {TABLE_NAME}
                    ORDER BY timestamp DESC
                    LIMIT 1000
                ) ORDER BY timestamp ASC
                """,
                conn
            )

            if df.empty:
                raise HTTPException(status_code=404, detail="Нет данных в таблице")

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Ошибка SQLite: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки данных: {str(e)}")

# Загрузка модели
print("Загрузка модели TimesFM...")
try:
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="pytorch",
            per_core_batch_size=32,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    print("Модель загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели TimesFM: {str(e)}")
    raise

# Эндпоинт для предсказания
@app.get("/predict")
def predict():
    try:
        print("Запрос на прогнозирование BTC/USDT, таймфрейм 1d")

        df = load_data()
        print(f"Загружено {len(df)} дневных записей. Последняя: {df['timestamp'].iloc[-1]}")

        if df["close"].isnull().any():
            df["close"] = df["close"].interpolate(method="linear")

        if df["close"].isnull().any():
            raise ValueError("После интерполяции остались пропущенные значения в 'close'.")

        forecast_input = [df["close"].values]
        frequency_input = [1]  # дневные данные

        print("Начинаем прогнозирование")
        point_forecast, _ = tfm.forecast(forecast_input, freq=frequency_input)

        forecast_df = pd.DataFrame({
            "timestamp": pd.date_range(
                start=df["timestamp"].iloc[-1] + pd.Timedelta(days=1),
                periods=len(point_forecast[0]),
                freq="D"
            ),
            "point_forecast": point_forecast[0],
        })

        print("Прогноз завершён")

        return {
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "last_historical_timestamp": df["timestamp"].iloc[-1].isoformat(),
            "last_historical_price": float(df["close"].iloc[-1]),
            "forecast": forecast_df.to_dict(orient="records")
        }

    except HTTPException as e:
        print(f"HTTP ошибка: {e.detail}")
        raise e
    except Exception as e:
        print(f"Ошибка при прогнозировании: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при прогнозировании: {str(e)}")

# Запуск (для отладки)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
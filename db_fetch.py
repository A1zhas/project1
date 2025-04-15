import ccxt
import sqlite3
import time
from datetime import datetime

# Настройки
DB_PATH = "dataset/crypto_data.db"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1d"
EXCHANGE = ccxt.binance()

# Функция для загрузки данных
def fetch_binance_data(symbol, timeframe, since):
    all_data = []
    while since < time.time() * 1000:
        try:
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Следующая метка времени
            time.sleep(1)  # Пауза, чтобы избежать блокировки API
        except Exception as e:
            print("Ошибка загрузки данных:", e)
            break
    return all_data

# Функция для сохранения в SQLite
def save_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS btc_data (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    """)
    for row in data:
        cur.execute("""
            INSERT OR IGNORE INTO btc_data (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row)
    conn.commit()
    conn.close()

# Загрузка данных с самого начала торговли (2011-09-13)
start_timestamp = int(datetime(2011, 9, 13).timestamp()) * 1000
btc_data = fetch_binance_data(SYMBOL, TIMEFRAME, start_timestamp)
save_to_db(btc_data)
print(f"Загружено {len(btc_data)} записей в {DB_PATH}")

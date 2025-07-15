import streamlit as st
import ccxt
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích Kỹ thuật Trading",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề ứng dụng
st.title("📈 Phần mềm Phân tích Kỹ thuật Trading")
st.markdown("""
**Công cụ phân tích đa khung thời gian với các chỉ báo kỹ thuật nâng cao**
""")

# Sidebar điều khiển
with st.sidebar:
    st.header("Thiết lập")

    # Chọn sàn giao dịch
    exchange_name = st.selectbox(
        "Sàn giao dịch",
        ["binance", "bybit", "kucoin", "okx"],
        index=0
    )

    # Chọn cặp tiền
    symbol = st.text_input("Cặp tiền", "BTC/USDT").upper()

    # Chọn khung thời gian
    timeframe = st.selectbox(
        "Khung thời gian",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=4
    )

    # Chỉ báo kỹ thuật
    st.subheader("Chỉ báo kỹ thuật")
    show_rsi = st.checkbox("RSI", True)
    show_macd = st.checkbox("MACD", True)
    show_bollinger = st.checkbox("Bollinger Bands", True)
    show_ma = st.checkbox("Đường MA", True)
    show_ichimoku = st.checkbox("Ichimoku Cloud", False)

    # Cài đặt nâng cao
    with st.expander("Cài đặt nâng cao"):
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        bb_period = st.slider("Bollinger Bands Period", 10, 50, 20)
        ma_fast = st.slider("MA Nhanh", 5, 50, 9)
        ma_slow = st.slider("MA Chậm", 20, 200, 21)

@st.cache_data(ttl=60)
def get_ohlcv_data(exchange_name, symbol, timeframe, limit=100):
    try:
        exchange = getattr(ccxt, exchange_name)()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df):
    if show_rsi:
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
    if show_macd:
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
    if show_bollinger:
        bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
    if show_ma:
        df[f'MA_{ma_fast}'] = ta.trend.SMAIndicator(df['close'], window=ma_fast).sma_indicator()
        df[f'MA_{ma_slow}'] = ta.trend.SMAIndicator(df['close'], window=ma_slow).sma_indicator()
    if show_ichimoku:
        ichimoku = ta.trend.IchimokuIndicator(
            high=df['high'],
            low=df['low'],
            window1=9,
            window2=26,
            window3=52
        )
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
    return df

def detect_signals(df):
    signals = pd.DataFrame(index=df.index)
    signals['Buy'] = 0
    signals['Sell'] = 0

    if show_rsi:
        signals['Buy'] = ((df['RSI'] < 30) & (df['RSI'].shift(1) >= 30))
        signals['Sell'] = ((df['RSI'] > 70) & (df['RSI'].shift(1) <= 70))

    if show_macd:
        macd_cross_up = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        macd_cross_down = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        signals['Buy'] = signals['Buy'] | macd_cross_up
        signals['Sell'] = signals['Sell'] | macd_cross_down

    if show_bollinger:
        signals['Buy'] = signals['Buy'] | (df['close'] < df['BB_lower'])
        signals['Sell'] = signals['Sell'] | (df['close'] > df['BB_upper'])

    if show_ma:
        ma_cross_up = (df[f'MA_{ma_fast}'] > df[f'MA_{ma_slow}']) & (df[f'MA_{ma_fast}'].shift(1) <= df[f'MA_{ma_slow}'].shift(1))
        ma_cross_down = (df[f'MA_{ma_fast}'] < df[f'MA_{ma_slow}']) & (df[f'MA_{ma_fast}'].shift(1) >= df[f'MA_{ma_slow}'].shift(1))
        signals['Buy'] = signals['Buy'] | ma_cross_up
        signals['Sell'] = signals['Sell'] | ma_cross_down

    if show_ichimoku:
        price_above_cloud = (df['close'] > df['Ichimoku_SpanA']) & (df['close'] > df['Ichimoku_SpanB'])
        conversion_above_base = (df['Ichimoku_Conversion'] > df['Ichimoku_Base'])
        signals['Buy'] = signals['Buy'] | (price_above_cloud & conversion_above_base)

        price_below_cloud = (df['close'] < df['Ichimoku_SpanA']) & (df['close'] < df['Ichimoku_SpanB'])
        conversion_below_base = (df['Ichimoku_Conversion'] < df['Ichimoku_Base'])
        signals['Sell'] = signals['Sell'] | (price_below_cloud & conversion_below_base)

    return signals

def _get_signal_sources(signals, df, show_rsi, show_macd, show_bollinger, show_ma, show_ichimoku, ma_fast, ma_slow):
    sources = []
    for idx in signals.index:
        source = []
        if show_rsi and (df.loc[idx, 'RSI'] < 30 or df.loc[idx, 'RSI'] > 70):
            source.append("RSI")
        if show_macd and (
            (df.loc[idx, 'MACD'] > df.loc[idx, 'MACD_Signal'] and df.shift(1).loc[idx, 'MACD'] <= df.shift(1).loc[idx, 'MACD_Signal']) or
            (df.loc[idx, 'MACD'] < df.loc[idx, 'MACD_Signal'] and df.shift(1).loc[idx, 'MACD'] >= df.shift(1).loc[idx, 'MACD_Signal'])
        ):
            source.append("MACD")
        if show_bollinger and (
            df.loc[idx, 'close'] < df.loc[idx, 'BB_lower'] or df.loc[idx, 'close'] > df.loc[idx, 'BB_upper']
        ):
            source.append("Bollinger")
        if show_ma and (
            (df.loc[idx, f'MA_{ma_fast}'] > df.loc[idx, f'MA_{ma_slow}'] and df.shift(1).loc[idx, f'MA_{ma_fast}'] <= df.shift(1).loc[idx, f'MA_{ma_slow}']) or
            (df.loc[idx, f'MA_{ma_fast}'] < df.loc[idx, f'MA_{ma_slow}'] and df.shift(1).loc[idx, f'MA_{ma_fast}'] >= df.shift(1).loc[idx, f'MA_{ma_slow}'])
        ):
            source.append("MA Cross")
        if show_ichimoku:
            if (df.loc[idx, 'close'] > df.loc[idx, 'Ichimoku_SpanA'] and
                df.loc[idx, 'close'] > df.loc[idx, 'Ichimoku_SpanB'] and
                df.loc[idx, 'Ichimoku_Conversion'] > df.loc[idx, 'Ichimoku_Base']):
                source.append("Ichimoku Bullish")
            elif (df.loc[idx, 'close'] < df.loc[idx, 'Ichimoku_SpanA'] and
                  df.loc[idx, 'close'] < df.loc[idx, 'Ichimoku_SpanB'] and
                  df.loc[idx, 'Ichimoku_Conversion'] < df.loc[idx, 'Ichimoku_Base']):
                source.append("Ichimoku Bearish")
        sources.append(", ".join(source))
    return sources

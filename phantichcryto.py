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
        ["binance", "bybit", "kucoin", "okx", "ONUS"],
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
def get_ohlcv_data(_exchange, symbol, timeframe, limit=100):
    try:
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.set_index('timestamp')
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df):
    # RSI
    if show_rsi:
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
    
    # MACD
    if show_macd:
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    if show_bollinger:
        bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
    
    # Moving Averages
    if show_ma:
        df[f'MA_{ma_fast}'] = ta.trend.SMAIndicator(df['close'], window=ma_fast).sma_indicator()
        df[f'MA_{ma_slow}'] = ta.trend.SMAIndicator(df['close'], window=ma_slow).sma_indicator()
    
    # Ichimoku Cloud
    if show_ichimoku:
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
    
    return df

def detect_signals(df):
    signals = pd.DataFrame(index=df.index)
    signals['Buy'] = 0
    signals['Sell'] = 0
    
    # Tín hiệu RSI
    if show_rsi:
        signals['Buy'] = ((df['RSI'] < 30) & (df['RSI'].shift(1) >= 30))
        signals['Sell'] = ((df['RSI'] > 70) & (df['RSI'].shift(1) <= 70))
    
    # Tín hiệu MACD
    if show_macd:
        macd_cross_up = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        macd_cross_down = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        signals['Buy'] = signals['Buy'] | macd_cross_up
        signals['Sell'] = signals['Sell'] | macd_cross_down
    
    # Tín hiệu Bollinger Bands
    if show_bollinger:
        signals['Buy'] = signals['Buy'] | (df['close'] < df['BB_lower'])
        signals['Sell'] = signals['Sell'] | (df['close'] > df['BB_upper'])
    
    # Tín hiệu MA Cross
    if show_ma:
        ma_cross_up = (df[f'MA_{ma_fast}'] > df[f'MA_{ma_slow}']) & (df[f'MA_{ma_fast}'].shift(1) <= df[f'MA_{ma_slow}'].shift(1))
        ma_cross_down = (df[f'MA_{ma_fast}'] < df[f'MA_{ma_slow}']) & (df[f'MA_{ma_slow}'].shift(1) >= df[f'MA_{ma_slow}'].shift(1))
        signals['Buy'] = signals['Buy'] | ma_cross_up
        signals['Sell'] = signals['Sell'] | ma_cross_down
    
    # Tín hiệu Ichimoku
    if show_ichimoku:
        price_above_cloud = (df['close'] > df['Ichimoku_SpanA']) & (df['close'] > df['Ichimoku_SpanB'])
        conversion_above_base = (df['Ichimoku_Conversion'] > df['Ichimoku_Base'])
        signals['Buy'] = signals['Buy'] | (price_above_cloud & conversion_above_base)
        
        price_below_cloud = (df['close'] < df['Ichimoku_SpanA']) & (df['close'] < df['Ichimoku_SpanB'])
        conversion_below_base = (df['Ichimoku_Conversion'] < df['Ichimoku_Base'])
        signals['Sell'] = signals['Sell'] | (price_below_cloud & conversion_below_base)
    
    return signals

def _get_signal_sources(signal_indexes, df):
    sources = []
    for idx in signal_indexes:
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

def plot_main_chart(df, signals):
    fig = go.Figure()
    
    # Vẽ nến
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Giá'
    ))
    
    # Bollinger Bands
    if show_bollinger:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            line=dict(color='rgba(200, 200, 200, 0.7)'),
            name='BB Upper',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_middle'],
            line=dict(color='rgba(150, 150, 150, 0.7)'),
            name='BB Middle',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            line=dict(color='rgba(200, 200, 200, 0.7)'),
            name='BB Lower',
            hoverinfo='skip',
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.1)'
        ))
    
    # Moving Averages
    if show_ma:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'MA_{ma_fast}'],
            line=dict(color='blue', width=1),
            name=f'MA {ma_fast}'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[f'MA_{ma_slow}'],
            line=dict(color='orange', width=1),
            name=f'MA {ma_slow}'
        ))
    
    # Ichimoku Cloud
    if show_ichimoku:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Ichimoku_SpanA'],
            line=dict(color='rgba(0, 0, 0, 0)'),
            name='Ichimoku Span A',
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Ichimoku_SpanB'],
            line=dict(color='rgba(0, 0, 0, 0)'),
            name='Ichimoku Span B',
            hoverinfo='skip',
            fill='tonexty',
            fillcolor='rgba(100, 100, 200, 0.2)'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Ichimoku_Conversion'],
            line=dict(color='green', width=1),
            name='Conversion Line'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Ichimoku_Base'],
            line=dict(color='red', width=1),
            name='Base Line'
        ))
    
    # Tín hiệu mua/bán
    buy_signals = signals[signals['Buy'] == True]
    sell_signals = signals[signals['Sell'] == True]
    
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=df.loc[buy_signals.index, 'low'] * 0.99,
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green'
            ),
            name='Tín hiệu Mua'
        ))
    
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=df.loc[sell_signals.index, 'high'] * 1.01,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red'
            ),
            name='Tín hiệu Bán'
        ))
    
    # Cấu hình layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_indicator_charts(df):
    fig = go.Figure()
    
    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            line=dict(color='purple'),
            name='RSI'
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    
    # MACD
    if show_macd:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='MACD Hist',
            marker_color=np.where(df['MACD_Hist'] < 0, 'red', 'green')
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            line=dict(color='blue'),
            name='MACD'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            line=dict(color='orange'),
            name='Signal'
        ))
    
    # Cấu hình layout
    fig.update_layout(
        height=300,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# Khởi tạo exchange
exchange = getattr(ccxt, exchange_name)({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

# Main App
df = get_ohlcv_data(exchange, symbol, timeframe)
if not df.empty:
    df = calculate_indicators(df)
    signals = detect_signals(df)
    
    # Hiển thị biểu đồ
    st.plotly_chart(plot_main_chart(df, signals), use_container_width=True)
    st.plotly_chart(plot_indicator_charts(df), use_container_width=True)
    
    # Hiển thị tín hiệu giao dịch
    with st.expander("Chi tiết tín hiệu giao dịch", expanded=True):
        st.write("**Tín hiệu Mua**")
        buy_signals = signals[signals['Buy'] == True]
        if not buy_signals.empty:
            buy_details = pd.DataFrame({
                'Thời gian': buy_signals.index,
                'Giá': df.loc[buy_signals.index, 'close'],
                'Chỉ báo': _get_signal_sources(buy_signals.index, df)
            })
            st.dataframe(buy_details)
        else:
            st.warning("Không có tín hiệu mua trong khung thời gian này")
        
        st.write("**Tín hiệu Bán**")
        sell_signals = signals[signals['Sell'] == True]
        if not sell_signals.empty:
            sell_details = pd.DataFrame({
                'Thời gian': sell_signals.index,
                'Giá': df.loc[sell_signals.index, 'close'],
                'Chỉ báo': _get_signal_sources(sell_signals.index, df)
            })
            st.dataframe(sell_details)
        else:
            st.warning("Không có tín hiệu bán trong khung thời gian này")
    
    # Hiển thị dữ liệu thô
    with st.expander("Xem dữ liệu thô"):
        st.dataframe(df.tail(20))
else:
    st.error("Không thể lấy dữ liệu. Vui lòng kiểm tra lại cặp tiền hoặc sàn giao dịch")

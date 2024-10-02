# Tony's Code

# Chameleon Indicators breakdown:
# Under the hood, Trend Chameleon evaluates four conditions to provide a directional strength score:
# 1. Whether the MACD value is positive.
# 2. Whether the SMA 50 of open prices is above the SMA 50 of the close prices.
# 3. Whether the ROC indicator value is positive.
# 4. Whether the current close price is above the SMA 50.
# The total number of fulfilled conditions (0 to 4) determines the trend strength, with 0 indicating the most bearish and 4 signifying the strongest bullish trend. 
# This score is then visually represented by coloring the bars on the chart.
# pip install numpy talib matplotlib

import talib
import numpy as np
import matplotlib.pyplot as plt

# Define input parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_LEN = 50
ROC_LEN = 25

# Define colors
colors = {
    0: 'purple',  # Very Bearish
    1: 'red',     # Bearish
    2: 'yellow',  # Neutral
    3: 'green',   # Bullish
    4: 'teal'     # Very Bullish
}

# Sample data (replace with your actual data)
close = np.random.random(100) * 100
open = np.random.random(100) * 100
high = np.random.random(100) * 100
low = np.random.random(100) * 100

# Calculate indicators
macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL)
sma_open = talib.SMA(open, timeperiod=SMA_LEN)
sma_close = talib.SMA(close, timeperiod=SMA_LEN)
roc = talib.ROC(close, timeperiod=ROC_LEN)

# Determine color based on conditions
v1 = 1 if macd[-1] > 0 else 0
v2 = 1 if sma_open[-1] > sma_close[-1] else 0
v3 = 1 if roc[-1] > 0 else 0
v4 = 1 if close[-1] > sma_close[-1] else 0

color_index = v1 + v2 + v3 + v4
c = colors.get(color_index, 'black')  # Default to black if index is out of range

# Plotting
plt.figure(figsize=(10, 5))
for i in range(len(close)):
    plt.plot([i, i], [low[i], high[i]], color=c)  # Wick
    plt.plot([i, i], [open[i], close[i]], color=c, linewidth=6)  # Candle body

plt.title('Trend Chameleon')
plt.show()

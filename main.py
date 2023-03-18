import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


START = "2015-01-01"
TODAY = pd.to_datetime("today").strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Select Stock
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME','ADANIENT.NS','^NSEI')
selected_stock = st.selectbox('Select stock for prediction:', stocks)

# Select prediction days
n_days = st.slider('Days of prediction:', 1, 30)
period = n_days

# Load stock data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Plot raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", mode='lines'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", mode='lines'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Predict and plot forecast
st.subheader("Stock Forecast")
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = pd.DatetimeIndex(df_train['ds']).tz_localize(None)

m = Prophet(interval_width=0.95)
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(f'Forecast plot for {n_days} days')
forecast_subset = forecast[-period:]
fig = plot_plotly(m, forecast_subset)
fig.update_traces(mode='lines')
st.plotly_chart(fig)

# Show forecast components
if st.checkbox("Show forecast components"):
    st.subheader("Forecast Components")
    fig = plot_components_plotly(m, forecast)
    fig.update_traces(mode='lines')
    st.plotly_chart(fig)
    
    
# Calculate evaluation metrics
y_true = data['Close'][-period:]
y_pred = forecast['yhat'][-period:].values
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

# Display evaluation metrics
st.write(f'RMSE: {rmse:.2f}')
st.write(f'MAE: {mae:.2f}')

# Plot actual vs predicted
def plot_actual_vs_predicted():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true.index, y=y_true, name="Actual"))
    fig.add_trace(go.Scatter(x=y_true.index, y=y_pred, name="Predicted"))
    fig.update_layout(
        title_text='Actual vs Predicted Stock Price',
        xaxis_title='Date',
        yaxis_title='Stock Price'
    )
    st.plotly_chart(fig)
st.subheader("Actual vs Predicted")
plot_actual_vs_predicted()
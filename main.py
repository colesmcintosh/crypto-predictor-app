import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.set_page_config(page_title='Stock Predictions', layout = 'wide', initial_sidebar_state = 'auto')

st.title("Stock Predictions")

stocks = ("AAPL", "GOOG", "TSLA", "MSFT", "GME")
selected_stock = st.selectbox("Select stock", stocks, format_func=lambda stock: f"{yf.Ticker(stock).info['longName']} ({stock})")

n_years = st.slider("Years of predication:", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data has loaded.")

st.subheader(f"{selected_stock} data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text="Actual Prices", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
fig1.layout.update(title_text="Prediction Prices", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
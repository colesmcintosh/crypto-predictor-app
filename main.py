import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

def main():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    st.set_page_config(page_title='Crypto Predictions', layout = 'wide', initial_sidebar_state = 'auto')

    st.title("Crypto Predictions")

    cryptos = (
    "ETH-USD",
    "BTC-USD",
    "FIL-USD",
    "XTZ-USD",
    "ETC-USD",
    "MANA-USD",
    "BCH-USD",
    "LINK-USD",
    "LTC-USD",
    "CRO-USD",
    "MATIC-USD",
    "SHIB-USD",
    "DOGE-USD",
    "ADA-USD",
    "XRP-USD",
    "USDC-USD",
    "BNB-USD",
    "USDT-USD")

    selected_crypto = st.selectbox("Select crypto", cryptos, format_func=lambda crypto: f"{yf.Ticker(crypto).info['name']} ({crypto})")

    n_years = st.slider("Years of predication:", 1, 5)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        
        return data

    data_load_state = st.text("Loading data...")
    data = load_data(selected_crypto)
    data_load_state.text("Data has loaded.")

    st.subheader(f"{selected_crypto} data")
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
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

if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import ta
import joblib
from configparser import ConfigParser
import oandapyV20
import oandapyV20.exceptions
import requests.exceptions
from oandapyV20.endpoints.instruments import InstrumentsCandles
import matplotlib.pyplot as plt


# --- Load OANDA credentials from Streamlit secrets ---
@st.cache_resource
def load_oanda_config():
    try:
        account_id = st.secrets["oanda"]["account_id"]
        access_token = st.secrets["oanda"]["api_key"]
        return account_id, access_token
    except Exception as e:
        st.error("Missing credentials in secrets.toml.")
        return None, None

account_id, access_token = load_oanda_config()

client = None
if access_token:
    try:
        client = oandapyV20.API(access_token=access_token)
        st.success("OANDA API client instantiated.")
    except Exception as e:
        st.error(f"Error instantiating OANDA API client: {e}")

# --- Data Fetching Function (using OANDA) ---
def fetch_latest_data_oanda(pair_symbol, count=500, granularity='H1'):
    """
    Fetches the most recent data for a given currency pair from OANDA.

    Args:
        pair_symbol: The symbol of the currency currency pair (e.g., 'USD_ZAR').
        count: The number of data points to fetch.
        granularity: The data interval (e.g., 'H1' for 1 hour, 'M15' for 15 minutes).

    Returns:
        A pandas DataFrame containing the fetched data or an empty DataFrame if an error occurs.
    """
    if not client:
        st.warning("OANDA API client is not initialized. Cannot fetch data.")
        return pd.DataFrame()

    st.info(f"Fetching latest data for {pair_symbol} from OANDA...")

    params = {
        "count": count,
        "granularity": granularity
    }

    try:
        r = InstrumentsCandles(instrument=pair_symbol, params=params)
        client.request(r)
        data = r.response.get('candles')

        if not data:
            st.warning(f"No candle data received for {pair_symbol}")
            return pd.DataFrame()

        # Process the response into a DataFrame
        df = pd.DataFrame(data)

        # Extract and rename columns
        df = df[['time', 'volume', 'mid']]
        df = pd.json_normalize(df['mid']).add_prefix('mid_')
        df = df.join(pd.DataFrame(data)[['time', 'volume']])

        # Rename columns to match the original code's expectations
        df.rename(columns={
            'mid_o': 'Open',
            'mid_h': 'High',
            'mid_l': 'Low',
            'mid_c': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Convert 'time' to datetime and set as index
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # Ensure column order matches expectations (optional but good practice)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        if df.empty:
            st.warning(f"No data fetched for {pair_symbol}")

        return df

    except oandapyV20.exceptions.V20Error as e:
        st.error(f"OANDA V20Error: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"RequestException during OANDA API call: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data: {e}")
        return pd.DataFrame()


# --- Model Loading Function ---
def load_model(filename):
    """Loads a machine learning model from a file."""
    try:
        loaded_model = joblib.load(filename)
        st.success(f"Successfully loaded model from {filename}")
        return loaded_model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {filename}")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Signal Generation Function ---
def generate_trading_signal(loaded_model, df):
    """
    Generates a trading signal based on the latest data and a loaded model.

    Args:
        loaded_model: The trained machine learning model.
        df: A pandas DataFrame containing recent data for the currency pair.

    Returns:
        A string indicating the trading signal ('Buy', 'Sell', or 'Hold').
    """
    if df.empty:
        return "Hold (No data)"

    # Ensure the DataFrame has the necessary columns for indicators
    required_cols = ['Close', 'High', 'Low', 'Open']
    if not all(col in df.columns for col in required_cols):
         return "Hold (Missing data columns)"

    # Calculate technical indicators
    close_series = df['Close'].squeeze()

    df['sma'] = ta.trend.sma_indicator(close_series, window=10)
    df['rsi'] = ta.momentum.rsi(close_series, window=14)
    df['macd'] = ta.trend.macd_diff(close_series)
    df['ema'] = ta.trend.ema_indicator(close_series, window=10)
    bollinger = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2)
    df['bb_bbm'] = bollinger.bollinger_mavg()
    df['bb_bbh'] = bollinger.bollinger_hband()
    df['bb_bbl'] = bollinger.bollinger_lband()

    # Drop rows with NaN values created by indicator calculations
    df.dropna(inplace=True)

    if df.empty:
        return "Hold (Data insufficient after indicators)"

    # Select the latest data point for prediction
    latest_data = df.iloc[-1]

    # Select features used for training
    features = ['sma', 'rsi', 'macd', 'ema', 'bb_bbm', 'bb_bbh', 'bb_bbl']
    X_latest = latest_data[features].values.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(X_latest)[0]

    # Interpret the prediction as a trading signal
    if prediction == 1:
        return "Buy"
    else:
        return "Sell" # The model was trained to predict >0 return (buy) or <=0 return (sell/hold)


# --- Streamlit App Layout ---
st.title("Forex Trading Strategy Application")

st.write("This application uses a trained machine learning model to generate trading signals for selected currency pairs based on data from OANDA.")

# Currency pair selection (using OANDA format)
oanda_pairs = ['USD_ZAR', 'GBP_JPY', 'EUR_TRY', 'USD_TRY']
selected_pair = st.selectbox("Select Currency Pair:", oanda_pairs)

# Model loading (adjust filename based on how you saved your models)
# Assuming models were saved with filenames like best_gb_model_USDZAR.pkl
model_filename = f"best_gb_model_{selected_pair.replace('_', '')}.pkl"
loaded_model = load_model(model_filename)

if loaded_model:
    # Fetch latest data for the selected pair from OANDA
    latest_data_df = fetch_latest_data_oanda(selected_pair, count=100, granularity='H1') # Fetch 100 H1 candles

    if not latest_data_df.empty:
        # Generate trading signal
        trading_signal = generate_trading_signal(loaded_model, latest_data_df)

        st.subheader("Trading Signal:")
        if trading_signal == "Buy":
            st.success(f"Signal for {selected_pair}: {trading_signal}")
        elif trading_signal == "Sell":
            st.error(f"Signal for {selected_pair}: {trading_signal}")
        else:
            st.info(f"Signal for {selected_pair}: {trading_signal}")

        # Display latest data (optional)
        st.subheader("Latest Data:")
        st.dataframe(latest_data_df.tail())

        # Optional: Add a simple price chart
        st.subheader("Price Chart (Last 100 Candles):")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(latest_data_df.index, latest_data_df['Close'], label='Close Price')
        ax.set_title(f"{selected_pair} Price Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)


    else:
        st.warning("Could not fetch data to generate signal.")
else:
    st.warning("Model not loaded. Please ensure the model file exists and the OANDA configuration is correct.")

# You can add more features here, such as:
# - Displaying technical indicators
# - Backtesting results (if you integrate a backtesting engine)
# - User inputs for strategy parameters
# - Integration with a trading platform for live trading (requires significant additional development and risk management)

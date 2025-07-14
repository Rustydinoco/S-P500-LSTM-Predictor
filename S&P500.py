import torch
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import traceback
import joblib
from sklearn.preprocessing import MinMaxScaler
import time


class LSTMModel(nn.Module):
  def __init__(self, input_size = 1, hidden_layer_size  = 100, output= 1):
    super().__init__()
    self.hidden_layer_size = hidden_layer_size
    self.lstm = nn.LSTM(input_size, hidden_layer_size)
    self.linear = nn.Linear(hidden_layer_size, output)

  def forward(self, input_seq):
    lstm_out, _ = self.lstm(input_seq)
    prediction = self.linear(lstm_out[:, -1, :])
    return prediction


@st.cache_resource
def load_data(ticker):
  df = yf.download(ticker, start="2010-01-01")
  df.dropna(inplace = True)

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df.index = pd.to_datetime(df.index)
  return df

@st.cache_resource
def load_model_and_scaler():
  try:
      model = LSTMModel()
      model.load_state_dict(torch.load('model_LSTM_S&P500.pth'))
      model.eval()

      scaler = joblib.load('MinMaxScaler.joblib')
      return model, scaler

  except Exception as e:
      st.error(f"Error : {e}")
      st.text(traceback.format_exc())
      raise e


st.title("S&P 500 Stock Prediction")

with st.spinner("Memuat Data dan Model......"):
  time.sleep(3)
  data = load_data("^GSPC")
  model, scaler = load_model_and_scaler()

if model is None or scaler is None:
  st.stop()
  
tab1,tab2,tab3 = st.tabs(["Data Historis","Grafik","Prediksi"])

with tab1:
  st.subheader("Historical Data S&P 500")
  st.dataframe(data.tail(100))  

with tab2:
  if model is not None:
    st.subheader("Close Price S&P 500")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= data.index, y= data["Close"], name= "S&P 500"))
    fig.layout.update(title_text = "S&P 500 Price Movement", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
  


with tab3:
  st.subheader("Hasil Prediksi Model")
  if st.sidebar.button("Prediksi Harga Selanjutnya"):
    look_back = 60
    last_60_days = data["Close"].values[-look_back:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1,1))
    X_pred = torch.FloatTensor(last_60_days_scaled).unsqueeze(0)
    with torch.no_grad():
      prediction_scaled = model(X_pred)

    prediction_actual = scaler.inverse_transform(prediction_scaled.numpy().reshape(-1,1))

    last_date = data.index[-1]
    next_date = last_date + pd.Timedelta(days= 1)

    st.session_state["next_date"] = next_date
    st.session_state["prediction_actual"] = prediction_actual[0][0] 
    
  if "next_date" in st.session_state and "prediction_actual" in st.session_state:
    st.sidebar.metric(
          label=f"Prediksi Harga Tutup untuk { st.session_state.next_date.strftime('%Y-%m-%d')}",
          value=f"${ st.session_state.prediction_actual[0][0]:,.2f}"
      )
    st.sidebar.header("Prediksi Harga")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x= data.index, y= data["Close"], name= "S&P 500"))        
    fig.add_trace(go.Scatter(x= [next_date], y= [prediction_actual[0][0]], name= "Hasil Prerdiksi",mode ="markers", marker = dict(color = "red", size = 10,symbol ="star")))
    fig.layout.update(title_text = "S&P 500 Price Movement", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
                    
st.sidebar.info("Disclaimer: Ini adalah proyek teknis, bukan saran finansial.")

"""alpha_vantage_news.py - Alpha Vantage News and Sentiment Fetcher"""
import pandas as pd
import requests
import os

class AlphaVantageNews:
    BASE_URL = "https://www.alphavantage.co/query"
    def __init__(self, api_key: str):
        self.api_key = api_key
    def fetch_news_sentiment(self, start_date, end_date):
        print("Fetching News Sentiment from Alpha Vantage...")
        # FIX-INST: Use EURUSD instead of EUR/USD (API rejects slash)
        params = {"function": "NEWS_SENTIMENT", "tickers": "EURUSD", "time_from": start_date, "time_to": end_date, "sort": "EARLIEST", "apikey": self.api_key}
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=60)
            if response.status_code != 200: return pd.DataFrame()
            data = response.json()
            if "feed" not in data: print("  No feed found."); return pd.DataFrame()
            df = pd.DataFrame(data["feed"])
            df["datetime"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
            df.set_index("datetime", inplace=True)
            df["News_Sentiment_Score"] = pd.to_numeric(df["overall_sentiment_score"], errors="coerce")
            daily_df = df[["News_Sentiment_Score"]].resample("D").mean()
            daily_df["News_Volume"] = df["News_Sentiment_Score"].resample("D").count()
            daily_df.ffill(inplace=True)
            daily_df["Sentiment_MA7"] = daily_df["News_Sentiment_Score"].rolling(7).mean()
            print(f"  Fetched {len(df)} articles. Aggregated to {len(daily_df)} days.")
            return daily_df
        except Exception as e:
            print(f"  Error: {e}"); return pd.DataFrame()

if __name__ == "__main__":
    API_KEY = "HKV29AUFFRK26P61"
    fetcher = AlphaVantageNews(API_KEY)
    news_df = fetcher.fetch_news_sentiment("20230101T0000", "20240630T2359")
    if not news_df.empty:
        print("\nSample Data:"); print(news_df.tail())
        os.makedirs("data", exist_ok=True)
        news_df.to_csv("data/av_news_data.csv")
        print("\nSaved to data/av_news_data.csv")

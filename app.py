import streamlit as st
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from itertools import combinations

# --- שימוש ב-Cache של Streamlit לטעינת המודל ---
# הפונקציה הזו תרוץ רק פעם אחת, והתוצאה תישמר בזיכרון המטמון
@st.cache_resource
def load_nlp_model():
    print("טוען מודל NLP בפעם הראשונה...")
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- פונקציות איסוף נתונים (עם Cache לשיפור ביצועים) ---
@st.cache_data(ttl=60) # שמירת הנתונים במטמון למשך 60 שניות
def get_markets_data():
    """ מאחד את איסוף הנתונים מכל הפלטפורמות לפונקציה אחת """
    all_markets = []
    
    # Kalshi
    try:
        r = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets?status=open&limit=1000", timeout=10 )
        r.raise_for_status()
        kalshi_markets = r.json().get('markets', [])
        for m in kalshi_markets:
            if m.get('yes_price'):
                all_markets.append({'platform': 'Kalshi', 'title': m['title'], 'yes_price': m['yes_price'], 'no_price': 1.0 - m['yes_price']})
    except requests.RequestException as e:
        st.error(f"שגיאה בגישה ל-Kalshi: {e}")

    # Polymarket
    try:
        r = requests.get("https://gamma-api.polymarket.com/events?closed=false&limit=1000", timeout=10 )
        r.raise_for_status()
        polymarket_events = r.json().get('data', [])
        for e in polymarket_events:
            for m in e.get('markets', []):
                if len(m.get('outcomes', [])) == 2:
                    all_markets.append({'platform': 'Polymarket', 'title': e.get('question'), 'yes_price': float(m['outcomes'][0]['price']), 'no_price': float(m['outcomes'][1]['price'])})
    except requests.RequestException as e:
        st.error(f"שגיאה בגישה ל-Polymarket: {e}")

    # Manifold Markets
    try:
        r = requests.get("https://api.manifold.markets/v0/markets?limit=1000", timeout=10 )
        r.raise_for_status()
        manifold_markets = r.json()
        for m in manifold_markets:
            # נתמקד בשווקים בינאריים (YES/NO) שעדיין פתוחים
            if m.get('outcomeType') == 'BINARY' and not m.get('isReso

import streamlit as st
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from itertools import combinations

# --- שימוש ב-Cache של Streamlit לטעינת המודל ---
@st.cache_resource
def load_nlp_model():
    """טוען את מודל השפה פעם אחת ושומר אותו בזיכרון המטמון."""
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
        kalshi_data = r.json()
        kalshi_markets = kalshi_data.get('markets', [])
        for m in kalshi_markets:
            if m.get('yes_price'):
                all_markets.append({'platform': 'Kalshi', 'title': m['title'], 'yes_price': m['yes_price'], 'no_price': 1.0 - m['yes_price']})
    except requests.RequestException as e:
        st.error(f"שגיאה בגישה ל-Kalshi: {e}")

    # Polymarket - *** לוגיקה חדשה ומשופרת ***
    try:
        r = requests.get("https://gamma-api.polymarket.com/events?closed=false&limit=1000", timeout=10 )
        r.raise_for_status()
        polymarket_response = r.json()
        
        # בדיקה אם התשובה היא מילון שמכיל מפתח 'data'
        if isinstance(polymarket_response, dict) and 'data' in polymarket_response:
            polymarket_events = polymarket_response.get('data', [])
        # בדיקה אם התשובה היא ישירות רשימה
        elif isinstance(polymarket_response, list):
            polymarket_events = polymarket_response
        else:
            polymarket_events = []
            st.warning("Polymarket API returned an unknown data format. Skipping for this scan.")

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
            if m.get('outcomeType') == 'BINARY' and not m.get('isResolved'):
                all_markets.append({'platform': 'Manifold', 'title': m['question'], 'yes_price': m['probability'], 'no_price': 1.0 - m['probability']})
    except requests.RequestException as e:
        st.error(f"שגיאה בגישה ל-Manifold: {e}")
        
    return all_markets

# --- פונקציית הליבה להשוואה ---
def find_all_opportunities(all_markets, model, similarity_thresh, profit_thresh):
    opportunities = []
    markets_by_platform = {p: [] for p in ['Kalshi', 'Polymarket', 'Manifold']}
    for market in all_markets:
        markets_by_platform[market['platform']].append(market)

    platform_pairs = combinations(markets_by_platform.keys(), 2)

    for p1_name, p2_name in platform_pairs:
        p1_markets = markets_by_platform[p1_name]
        p2_markets = markets_by_platform[p2_name]

        if not p1_markets or not p2_markets: continue

        p1_titles = [m['title'] for m in p1_markets]
        p2_titles = [m['title'] for m in p2_markets]

        p1_embeddings = model.encode(p1_titles, convert_to_tensor=True)
        p2_embeddings = model.encode(p2_titles, convert_to_tensor=True)
        cosine_scores = util.cos_sim(p1_embeddings, p2_embeddings)

        for i in range(len(p1_titles)):
            best_match_idx = np.argmax(cosine_scores[i])
            score = cosine_scores[i][best_match_idx]
            
            if score >= similarity_thresh:
                m1 = p1_markets[i]
                m2 = p2_markets[best_match_idx]
                
                cost1 = m1['yes_price'] + m2['no_price']
                if cost1 < 1.0:
                    roi1 = ((1.0 - cost1) / cost1) * 100
                    if roi1 >= profit_thresh:
                        opportunities.append([f"{p1_name} vs {p2_name}", f"{score:.2f}", m1['title'], m2['title'], f"Buy YES@{p1_name}, Buy NO@{p2_name}", f"{roi1:.2f}%"])

                cost2 = m2['yes_price'] + m1['no_price']
                if cost2 < 1.0:
                    roi2 = ((1.0 - cost2) / cost2) * 100
                    if roi2 >= profit_thresh:
                        opportunities.append([f"{p1_name} vs {p2_name}", f"{score:.2f}", m2['title'], m1['title'], f"Buy YES@{p2_name}, Buy NO@{p1_name}", f"{roi2:.2f}%"])
    
    return opportunities

# --- בניית ממשק ה-Streamlit ---
st.set_page_config(page_title="בוט ארביטראז' מרובה מקורות", layout="wide")
st.title("🤖 בוט ארביטראז' מרובה מקורות")
st.markdown("סורק את **Kalshi, Polymarket, ו-Manifold Markets** לאיתור הזדמנויות.")

st.sidebar.header("⚙️ הגדרות סריקה")
similarity_threshold = st.sidebar.slider("סף דמיון סמנטי", 0.7, 1.0, 0.85, 0.01)
profit_threshold = st.sidebar.slider("סף רווח נטו מינימלי (ROI %)", 0.0, 20.0, 1.0, 0.5)

if st.sidebar.button("🚀 סרוק עכשיו"):
    with st.spinner("טוען מודל שפה ושולף נתונים מכל הפלטפורמות..."):
        model = load_nlp_model()
        all_markets_data = get_markets_data()
        
        st.info(f"סה\"כ נשלפו {len(all_markets_data)} שווקים מכל הפלטפורמות.")
        
        results = find_all_opportunities(all_markets_data, model, similarity_threshold, profit_threshold)

    st.success("הסריקה הושלמה!")

    if results:
        st.subheader(f"נמצאו {len(results)} הזדמנויות ארביטראז':")
        df = pd.DataFrame(results, columns=["זוג פלטפורמות", "דמיון", "שוק 1", "שוק 2", "אסטרטגיה", "רווח (ROI)"])
        st.dataframe(df)
    else:
        st.warning("לא נמצאו הזדמנויות ארביטראז' העומדות בסף שהוגדר.")

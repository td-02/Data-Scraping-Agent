import streamlit as st
import pandas as pd
from business_agent import run_agent

st.title("🧭 Local Business Data Agent (Free)")
st.markdown("Enter your business data prompt below (e.g. *Find restaurants around South Kolkata within 2 km*)")

prompt = st.text_area("Prompt:", "Find cafes and bakeries in Ballygunge, Kolkata within 2 km")
btn = st.button("Run Agent")

if btn and prompt.strip():
    with st.spinner("Running agent..."):
        df = run_agent(prompt, "results.csv")
        st.success(f"Fetched {len(df)} results!")
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "results.csv", "text/csv")

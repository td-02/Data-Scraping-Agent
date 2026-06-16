"""Streamlit frontend for the deterministic business suitability pipeline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from business_suitability_single_file import evaluate_query


st.title("Local Business Suitability Agent")
st.markdown("Run the deterministic suitability pipeline and inspect the ranked output.")

prompt = st.text_area("Prompt", "Find cafes and bakeries in Ballygunge, Kolkata within 2 km")
pages = st.slider("Pages to fetch", min_value=1, max_value=3, value=1)
radius_m = st.slider("Local competitor radius (meters)", min_value=100, max_value=2000, value=500, step=100)
use_llm = st.checkbox("Use Anthropic verdicts", value=False)

if st.button("Run Agent") and prompt.strip():
    with st.spinner("Running suitability analysis..."):
        rows = evaluate_query(prompt.strip(), pages=pages, local_radius_m=radius_m, use_llm=use_llm)
        df = pd.DataFrame(rows)
        st.success(f"Evaluated {len(df)} businesses.")
        st.dataframe(df.head(20))
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "business_suitability_results.csv", "text/csv")

import streamlit as st
from src.inference import ScoringService

# Set Streamlit page config
st.set_page_config(page_title="Customer Query Intent Classifier", layout="centered")

# App title
st.title("🧠 Multi-Intent Classifier")
st.subheader("Classifies customer queries into multiple intents")

# Description
st.markdown("""
**Supported Intents:**
- Product Inquiry
- Order Tracking
- Refund Request
- Store Policy
""")

# Input box
query = st.text_area("Enter customer query", height=150)

# Button to classify
if st.button("Classify Intents"):
    if not query.strip():
        st.warning("Please enter a customer query.")
    else:
        with st.spinner("Classifying..."):
            intents = ScoringService.predict(query)
        if intents:
            st.success("Predicted Intents:")
            st.write(f"🎯 {', '.join(intents)}")
        else:
            st.info("No intents detected with confidence > 0.8.")

import streamlit as st
import pandas as pd
import plotly.express as px

# ========== SETTINGS ==========
st.set_page_config(
    page_title="Cybersecurity Risk Dashboard",
    page_icon="🔐",
    layout="wide"
)

# ========== HEADER ==========
st.sidebar.image("assets/logo.png", use_column_width=True)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Risk Register", "About"])

st.title("🔐 Cybersecurity Risk Dashboard")
st.markdown("A professional tool for monitoring and visualizing cybersecurity risks.")

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    return pd.read_csv("data/risk_register.csv")

df = load_data()

# ========== DASHBOARD ==========
if page == "Dashboard":
    st.subheader("📊 Risk Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Risks", len(df))
    col2.metric("Open Risks", len(df[df["Status"] == "Open"]))
    col3.metric("High Impact Risks", len(df[df["Impact"] == "High"]))

    # Pie chart: Risks by Status
    fig1 = px.pie(df, names="Status", title="Risks by Status")
    st.plotly_chart(fig1, use_container_width=True)

    # Bar chart: Risks by Owner
    fig2 = px.bar(df, x="Owner", color="Impact", title="Risks by Owner & Impact")
    st.plotly_chart(fig2, use_container_width=True)

# ========== RISK REGISTER ==========
elif page == "Risk Register":
    st.subheader("📋 Risk Register")
    st.dataframe(df, use_container_width=True)

    # Download option
    st.download_button(
        label="Download Risk Register (CSV)",
        data=df.to_csv(index=False),
        file_name="risk_register.csv",
        mime="text/csv"
    )

# ========== ABOUT ==========
else:
    st.subheader("ℹ️ About")
    st.markdown("""
    This dashboard demonstrates how cybersecurity professionals can monitor and visualize risks.  
    Features include:
    - Risk register management  
    - Interactive charts (Plotly)  
    - Export capability  

    **Built by Yasir Tasi’u Ado – Cybersecurity Professional**
    """)

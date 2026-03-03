import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
THEME_COLOUR = "#707275"
st.markdown(f"""
<style>

/* Reduce vertical spacing */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 1rem;
}}

/* Metric container */
div[data-testid="metric-container"] {{
    background-color: #F7F7F8;
    border: 1px solid #E0E0E0;
    padding: 10px 15px;
    border-radius: 8px;
}}

/* Metric label */
div[data-testid="metric-container"] label {{
    color: {THEME_COLOUR} !important;
    font-weight: 500;
    font-size: 0.85rem;
}}

/* Metric value (BIG number) */
div[data-testid="metric-container"] > div:nth-child(2) {{
    color: {THEME_COLOUR} !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
}}

/* Delta colour override (if you ever add delta values) */
div[data-testid="metric-container"] svg {{
    fill: {THEME_COLOUR} !important;
}}

/* Reduce gap between columns */
div[data-testid="column"] {{
    padding: 0.5rem;
}}

</style>
""", unsafe_allow_html=True)





st.markdown(f"<h1 style='color:{THEME_COLOUR};'>Hackathon</h1>", unsafe_allow_html=True)

st.write('This page looks at key analytics from Northstar Desk data') 



path = r"C:\Users\barnyrumbold\OneDrive - Kidney Research UK\Desktop\Hackathon\Northstar-Desk-Decision-Support-Tool-\Data\clean.csv"
data = pd.read_csv(path)
data['csat_score'] = pd.to_numeric(data['csat_score'], errors='coerce')

total_cases = len(data)
open_cases = len(data[data['status'] == 'open'])
avg_response = data['first_response_time_hours'].mean()
avg_resolution = data['resolution_time_hours'].mean()
avg_csat = data['csat_score'].mean()
escalation_rate = (data['escalated'] == True).mean() * 100

sla_met = (data['resolution_time_hours'] <= data['sla_target_hours']).mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Cases", total_cases)
col2.metric("Open Cases", open_cases)
col3.metric("% Within SLA", f"{sla_met:.1f}%")
col4.metric("Escalation Rate", f"{escalation_rate:.1f}%")
col1, col2 = st.columns(2, gap="small")

with col1:
    st.markdown("<h4 style='color:#707275;'>Case Type</h4>", unsafe_allow_html=True)
    
    case_type_counts = data['case_type'].value_counts().reset_index()
    case_type_counts.columns = ['case_type', 'count']

    fig1 = px.bar(
        case_type_counts,
        x='case_type',
        y='count',
        color_discrete_sequence=[THEME_COLOUR]
    )

    fig1.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=THEME_COLOUR,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig1, use_container_width=True)


with col2:
    st.markdown("<h4 style='color:#707275;'>Category</h4>", unsafe_allow_html=True)
    
    category_counts = data['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']

    fig2 = px.bar(
        category_counts,
        x='category',
        y='count',
        color_discrete_sequence=[THEME_COLOUR]
    )

    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color=THEME_COLOUR,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig2, use_container_width=True)
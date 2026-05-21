import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fiserv Effort Estimation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    stories = pd.read_excel("Fiserv_Dataset.xlsx", sheet_name="User_Stories")
    tasks   = pd.read_excel("Fiserv_Dataset.xlsx", sheet_name="Tasks")
    release = pd.read_excel("Fiserv_Dataset.xlsx", sheet_name="Release_Summary")
    df = stories.dropna(subset=["Actual_Hrs"]).copy()
    df["Bug_Rework_Hrs"] = df["Bug_Rework_Hrs"].fillna(0)
    df["Overrun_Pct_clean"] = (
        (df["Actual_Hrs"] - df["Original_Est_Hrs"])
        / df["Original_Est_Hrs"] * 100
    ).round(1)
    return df, tasks, release

df, tasks, release = load_data()

# ── Sidebar filters ───────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Fiserv_logo.svg/320px-Fiserv_logo.svg.png",
    width=180
)
st.sidebar.title("Filters")

releases  = ["All"] + sorted(df["Release_ID"].unique().tolist())
platforms = ["All"] + sorted(df["Platform"].dropna().unique().tolist())
itypes    = ["All"] + sorted(df["Item_Type"].unique().tolist())

sel_release  = st.sidebar.selectbox("Release",  releases)
sel_platform = st.sidebar.selectbox("Platform", platforms)
sel_itype    = st.sidebar.selectbox("Item Type", itypes)

filtered = df.copy()
if sel_release  != "All": filtered = filtered[filtered["Release_ID"] == sel_release]
if sel_platform != "All": filtered = filtered[filtered["Platform"]   == sel_platform]
if sel_itype    != "All": filtered = filtered[filtered["Item_Type"]  == sel_itype]

# ── Title ─────────────────────────────────────────────────────
st.title("📊 Fiserv Effort Estimation Dashboard")
st.caption("AI-Enabled Project Management via Microsoft Copilot — Exploratory Data Analysis")
st.divider()

# ── KPI cards ─────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total     = len(filtered)
avg_hrs   = filtered["Actual_Hrs"].mean()
med_hrs   = filtered["Actual_Hrs"].median()
over_pct  = (filtered["Overrun_Pct_clean"] > 0).sum() / len(filtered) * 100
defect_ov = filtered[filtered["Item_Type"] == "Defect"]["Overrun_Pct_clean"].mean()

k1.metric("Total Stories",       f"{total}")
k2.metric("Avg Actual Hours",    f"{avg_hrs:.1f} hrs")
k3.metric("Median Actual Hours", f"{med_hrs:.1f} hrs")
k4.metric("Stories Over Estimate", f"{over_pct:.0f}%")
k5.metric("Defect Avg Overrun",  f"+{defect_ov:.0f}%" if not np.isnan(defect_ov) else "N/A")

st.divider()

# ── Row 1: Overrun by Item Type + Distribution ────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Overrun % by Item Type")
    overrun_data = (
        filtered.groupby("Item_Type")["Overrun_Pct_clean"]
        .mean().reset_index()
        .rename(columns={"Overrun_Pct_clean": "Avg Overrun %"})
        .sort_values("Avg Overrun %", ascending=False)
    )
    color_map = {"Defect": "#D85A30", "Enhancement": "#BA7517", "Feature": "#2E75B6"}
    fig1 = px.bar(
        overrun_data,
        x="Item_Type",
        y="Avg Overrun %",
        color="Item_Type",
        color_discrete_map=color_map,
        text="Avg Overrun %",
        title="Average Overrun % by Work Item Type"
    )
    fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig1.update_layout(showlegend=False, height=380,
                       yaxis_title="Average Overrun %",
                       xaxis_title="Item Type")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Distribution of Actual Hours")
    fig2 = px.histogram(
        filtered,
        x="Actual_Hrs",
        nbins=40,
        color_discrete_sequence=["#2E75B6"],
        title="How Long Stories Actually Take",
        labels={"Actual_Hrs": "Actual Hours", "count": "Number of Stories"}
    )
    fig2.update_layout(height=380, bargap=0.05)
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Story Points scatter + Platform bar ────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Story Points vs Actual Hours")
    fig3 = px.scatter(
        filtered,
        x="Story_Points",
        y="Actual_Hrs",
        color="Item_Type",
        color_discrete_map=color_map,
        hover_data=["Epic", "Feature_Size", "Team_ID", "Original_Est_Hrs"],
        title="Complexity Score vs Time Taken",
        labels={"Story_Points": "Story Points",
                "Actual_Hrs": "Actual Hours",
                "Item_Type": "Work Type"},
        trendline="ols"
    )
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Average Hours by Platform")
    plat_data = (
        filtered.groupby("Platform")["Actual_Hrs"]
        .mean().reset_index()
        .rename(columns={"Actual_Hrs": "Avg Actual Hours"})
        .sort_values("Avg Actual Hours", ascending=False)
    )
    fig4 = px.bar(
        plat_data,
        x="Platform",
        y="Avg Actual Hours",
        color="Platform",
        color_discrete_sequence=["#1D9E75", "#2E75B6", "#7F77DD"],
        text="Avg Actual Hours",
        title="Which Platform Takes the Most Hours?"
    )
    fig4.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig4.update_layout(showlegend=False, height=420)
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Release comparison ──────────────────────────────────
st.subheader("Release-Level Comparison")
fig5 = go.Figure()
fig5.add_trace(go.Bar(
    name="Original Estimate",
    x=release["Release_ID"],
    y=release["Total_Orig_Est_Hrs"],
    marker_color="#B5D4F4"
))
fig5.add_trace(go.Bar(
    name="Actual Hours",
    x=release["Release_ID"],
    y=release["Total_Actual_Hrs"],
    marker_color="#D85A30"
))
fig5.update_layout(
    barmode="group",
    title="Estimated vs Actual Hours per Release",
    xaxis_title="Release",
    yaxis_title="Total Hours",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig5, use_container_width=True)

# ── Row 4: Correlation heatmap ─────────────────────────────────
st.subheader("Feature Correlation with Actual Hours")
col5, col6 = st.columns([2, 1])

with col5:
    numeric_cols = [
        "Story_Points", "Team_Size", "Sprint_Number",
        "Original_Est_Hrs", "Bug_Rework_Hrs", "N_Tasks", "Actual_Hrs"
    ]
    corr = filtered[numeric_cols].corr().round(2)
    fig6 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Correlation Matrix — Which Features Predict Hours?",
        aspect="auto"
    )
    fig6.update_layout(height=420)
    st.plotly_chart(fig6, use_container_width=True)

with col6:
    st.subheader("Top Predictors of Actual Hours")
    corr_target = (
        filtered[numeric_cols].corr()["Actual_Hrs"]
        .drop("Actual_Hrs").abs()
        .sort_values(ascending=False)
        .reset_index()
    )
    corr_target.columns = ["Feature", "Correlation"]
    fig7 = px.bar(
        corr_target,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale="Blues",
        title="Ranked by Predictive Strength",
        range_x=[0, 1]
    )
    fig7.update_layout(height=420, showlegend=False,
                       coloraxis_showscale=False)
    st.plotly_chart(fig7, use_container_width=True)

# ── Row 5: Overrun distribution ────────────────────────────────
st.subheader("Overrun Distribution")
col7, col8 = st.columns(2)

with col7:
    fig8 = px.histogram(
        filtered,
        x="Overrun_Pct_clean",
        nbins=40,
        color_discrete_sequence=["#7F77DD"],
        title="How Much Do Stories Overrun?",
        labels={"Overrun_Pct_clean": "Overrun % (positive = took longer)"}
    )
    fig8.add_vline(x=0, line_dash="dash", line_color="red",
                   annotation_text="Zero overrun", annotation_position="top right")
    fig8.update_layout(height=360)
    st.plotly_chart(fig8, use_container_width=True)

with col8:
    st.subheader("Stories by Item Type")
    type_counts = filtered["Item_Type"].value_counts().reset_index()
    type_counts.columns = ["Item Type", "Count"]
    fig9 = px.pie(
        type_counts,
        names="Item Type",
        values="Count",
        color="Item Type",
        color_discrete_map=color_map,
        title="Breakdown of Work Item Types",
        hole=0.4
    )
    fig9.update_layout(height=360)
    st.plotly_chart(fig9, use_container_width=True)

# ── Raw data table ─────────────────────────────────────────────
st.divider()
with st.expander("🔍 View Raw Data Table"):
    display_cols = [
        "Story_ID", "Release_ID", "Epic", "Item_Type", "Platform",
        "Story_Points", "Original_Est_Hrs", "Actual_Hrs",
        "Overrun_Pct_clean", "Status"
    ]
    st.dataframe(
        filtered[display_cols].rename(
            columns={"Overrun_Pct_clean": "Overrun %"}
        ),
        use_container_width=True,
        hide_index=True
    )

st.caption("Built by Mohor — University of Auckland Industry Placement | Fiserv Professional Services | April 2026")
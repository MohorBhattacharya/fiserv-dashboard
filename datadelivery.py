import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Fiserv Effort Estimation — AI Dashboard",
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

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Fiserv_logo.svg/320px-Fiserv_logo.svg.png",
    width=180
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "🤖 Copilot Assistant"])

color_map = {"Defect": "#D85A30", "Enhancement": "#BA7517", "Feature": "#2E75B6"}

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("📊 Fiserv Effort Estimation Dashboard")
    st.markdown("""
    > This dashboard analyses **435 completed User Stories** across 8 releases.
    > Use the filters on the left to drill into specific releases, platforms, or work types.
    > Every chart updates automatically when you change a filter.
    """)
    st.divider()

    # Filters
    st.sidebar.title("Filters")
    st.sidebar.markdown("*Narrow down the data shown in all charts below.*")
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

    if len(filtered) == 0:
        st.warning("⚠️ No stories match this filter combination. Try adjusting the filters.")
        st.stop()
    elif len(filtered) < 20:
        st.info(f"ℹ️ Only {len(filtered)} stories match this filter — results may not be fully representative.")

    # KPI cards
    total     = len(filtered)
    avg_hrs   = filtered["Actual_Hrs"].mean()
    med_hrs   = filtered["Actual_Hrs"].median()
    over_pct  = (filtered["Overrun_Pct_clean"] > 0).sum() / len(filtered) * 100
    defect_ov = filtered[filtered["Item_Type"] == "Defect"]["Overrun_Pct_clean"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Stories", f"{total}", help="Number of completed User Stories in this filter")
    k2.metric("Avg Actual Hours", f"{avg_hrs:.1f} hrs", help="Average hours a story actually took")
    k3.metric("Median Actual Hours", f"{med_hrs:.1f} hrs", help="Middle value — less affected by outliers than the average")
    k4.metric("Stories Over Estimate", f"{over_pct:.0f}%", help="Percentage of stories that took longer than originally estimated")
    k5.metric("Defect Avg Overrun", f"+{defect_ov:.0f}%" if not np.isnan(defect_ov) else "N/A", help="How much longer Defects took vs their estimate on average")

    st.divider()

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Overrun Analysis",
        "⏱️ Effort Patterns",
        "🔗 What Predicts Hours?",
        "📋 Raw Data"
    ])

    # ── TAB 1: Overrun Analysis ────────────────────────────────
    with tab1:
        st.markdown("### Why do projects run over time?")
        st.markdown("""
        These charts show the **estimation accuracy problem** — how often and by how much
        work takes longer than planned. The orange/red colours indicate overruns.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Overrun % by Work Type")
            st.markdown("*Bug fixes (Defects) consistently take nearly double their estimate. Features are the most predictable.*")
            overrun_data = (
                filtered.groupby("Item_Type")["Overrun_Pct_clean"]
                .mean().reset_index()
                .rename(columns={"Overrun_Pct_clean": "Avg Overrun %"})
                .sort_values("Avg Overrun %", ascending=False)
            )
            fig1 = px.bar(overrun_data, x="Item_Type", y="Avg Overrun %",
                          color="Item_Type", color_discrete_map=color_map,
                          text="Avg Overrun %")
            fig1.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig1.update_layout(showlegend=False, height=380,
                               yaxis_title="Average Overrun %", xaxis_title="Work Type",
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("#### How Much Do Individual Stories Overrun?")
            st.markdown("*The red line is zero — no overrun. Almost all bars are to the right, meaning most work takes longer than planned.*")
            fig8 = px.histogram(filtered, x="Overrun_Pct_clean", nbins=40,
                                color_discrete_sequence=["#7F77DD"],
                                labels={"Overrun_Pct_clean": "Overrun % (positive = took longer than estimated)"})
            fig8.add_vline(x=0, line_dash="dash", line_color="red",
                           annotation_text="← On time  |  Over estimate →",
                           annotation_position="top left")
            fig8.update_layout(height=380)
            st.plotly_chart(fig8, use_container_width=True)

        st.markdown("#### Release-by-Release: Estimated vs Actual Hours")
        st.markdown("*Light blue = what the team thought it would take. Orange = what it actually took. Orange is taller in every single release.*")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Original Estimate",
                              x=release["Release_ID"], y=release["Total_Orig_Est_Hrs"],
                              marker_color="#B5D4F4"))
        fig5.add_trace(go.Bar(name="Actual Hours",
                              x=release["Release_ID"], y=release["Total_Actual_Hrs"],
                              marker_color="#D85A30"))
        fig5.update_layout(barmode="group",
                           xaxis_title="Release", yaxis_title="Total Hours", height=400,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig5, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Breakdown by Work Type")
            st.markdown("*What proportion of work is Features, Enhancements, and Defects?*")
            type_counts = filtered["Item_Type"].value_counts().reset_index()
            type_counts.columns = ["Item Type", "Count"]
            fig9 = px.pie(type_counts, names="Item Type", values="Count",
                          color="Item Type", color_discrete_map=color_map, hole=0.4)
            fig9.update_layout(height=340)
            st.plotly_chart(fig9, use_container_width=True)

        with col4:
            st.markdown("#### Average Hours by Platform")
            st.markdown("*Does iOS, Android, or Backend work take different amounts of time?*")
            plat_data = (
                filtered.groupby("Platform")["Actual_Hrs"]
                .mean().reset_index()
                .rename(columns={"Actual_Hrs": "Avg Actual Hours"})
                .sort_values("Avg Actual Hours", ascending=False)
            )
            fig4 = px.bar(plat_data, x="Platform", y="Avg Actual Hours",
                          color="Platform",
                          color_discrete_sequence=["#1D9E75", "#2E75B6", "#7F77DD"],
                          text="Avg Actual Hours")
            fig4.update_traces(texttemplate="%{text:.1f} hrs", textposition="outside")
            fig4.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig4, use_container_width=True)

    # ── TAB 2: Effort Patterns ─────────────────────────────────
    with tab2:
        st.markdown("### How long does work actually take?")
        st.markdown("""
        These charts show the **distribution and patterns of actual effort**.
        Understanding these patterns is what powers the AI predictions in the Copilot Assistant.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribution of Actual Hours")
            st.markdown("*Most stories are quick (0–50 hrs). A few outliers take much longer. This is why the system gives a range, not a single number.*")
            fig2 = px.histogram(filtered, x="Actual_Hrs", nbins=40,
                                color_discrete_sequence=["#2E75B6"],
                                labels={"Actual_Hrs": "Actual Hours Taken", "count": "Number of Stories"})
            fig2.update_layout(height=380, bargap=0.05)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("#### Story Points vs Actual Hours")
            st.markdown("*Higher complexity (story points) consistently means more hours. The upward trend line proves it. Defects (orange) sit above the line — they take longer than their complexity suggests.*")
            fig3 = px.scatter(filtered, x="Story_Points", y="Actual_Hrs",
                              color="Item_Type", color_discrete_map=color_map,
                              hover_data=["Epic", "Feature_Size", "Team_ID", "Original_Est_Hrs"],
                              labels={"Story_Points": "Story Points (Complexity Score)",
                                      "Actual_Hrs": "Actual Hours Taken",
                                      "Item_Type": "Work Type"},
                              trendline="ols")
            fig3.update_layout(height=380)
            st.plotly_chart(fig3, use_container_width=True)

    # ── TAB 3: Predictors ──────────────────────────────────────
    with tab3:
        st.markdown("### What best predicts how long work will take?")
        st.markdown("""
        These charts show which **input signals** are most useful for predicting actual hours.
        The AI model uses these signals to generate estimates. Higher correlation = stronger predictor.
        """)

        numeric_cols = ["Story_Points", "Team_Size", "Sprint_Number",
                        "Original_Est_Hrs", "Bug_Rework_Hrs", "N_Tasks", "Actual_Hrs"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Top Predictors — Ranked")
            st.markdown("*Original estimate and story points are the strongest signals. Team size and sprint number add little.*")
            corr_target = (
                filtered[numeric_cols].corr()["Actual_Hrs"]
                .drop("Actual_Hrs").abs()
                .sort_values(ascending=False)
                .reset_index()
            )
            corr_target.columns = ["Feature", "Correlation"]
            fig7 = px.bar(corr_target, x="Correlation", y="Feature", orientation="h",
                          color="Correlation", color_continuous_scale="Blues",
                          range_x=[0, 1],
                          labels={"Correlation": "Predictive Strength (0 = none, 1 = perfect)"})
            fig7.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            st.markdown("#### Full Correlation Matrix")
            st.markdown("*Each square shows how strongly two columns are linked. Darker blue = stronger relationship. The bottom row shows what predicts Actual Hours.*")
            corr = filtered[numeric_cols].corr().round(2)
            fig6 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues",
                             aspect="auto")
            fig6.update_layout(height=380)
            st.plotly_chart(fig6, use_container_width=True)

    # ── TAB 4: Raw Data ────────────────────────────────────────
    with tab4:
        st.markdown("### Underlying Data")
        st.markdown("*Every row is one completed User Story. Scroll to explore. Use the filters on the left to narrow down.*")
        display_cols = ["Story_ID", "Release_ID", "Epic", "Item_Type", "Platform",
                        "Story_Points", "Original_Est_Hrs", "Actual_Hrs",
                        "Overrun_Pct_clean", "Status"]
        st.dataframe(
            filtered[display_cols].rename(columns={"Overrun_Pct_clean": "Overrun %"}),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Showing {len(filtered)} stories  |  Built by Mohor — University of Auckland Industry Placement | Fiserv Professional Services | April 2026")


# ═════════════════════════════════════════════════════════════
# PAGE 2 — COPILOT ASSISTANT
# ═════════════════════════════════════════════════════════════
elif page == "🤖 Copilot Assistant":

    st.title("🤖 Copilot Assistant")
    st.markdown("""
    > **How to use:** Fill in the work item details below, then type any question in the chat box.
    > The AI uses historical patterns from all past releases to estimate effort for any work item —
    > including new releases and new enhancements not yet in the dataset.
    """)
    st.divider()

    @st.cache_resource
    def load_models():
        import joblib
        import pickle
        xgb      = joblib.load("xgb_model.pkl")
        knn      = joblib.load("knn_model.pkl")
        ql       = joblib.load("qr_low.pkl")
        qm       = joblib.load("qr_mid.pkl")
        qh       = joblib.load("qr_high.pkl")
        shap_exp = joblib.load("shap_explainer.pkl")
        train_df = pd.read_csv("train_data.csv")
        with open("features.pkl", "rb") as f:
            feats = pickle.load(f)
        return xgb, knn, ql, qm, qh, shap_exp, train_df, feats

    xgb_m, knn_m, ql_m, qm_m, qh_m, shap_exp, train_df, FEATS = load_models()

    st.subheader("Step 1 — Describe the work item")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        existing_releases = sorted(df["Release_ID"].unique().tolist())
        release_options   = existing_releases + ["Future Release"]
        release_sel       = st.selectbox("Release", release_options,
                    help="Select an existing release or choose Future Release for upcoming releases. The model predicts for any release using cross-release patterns.")
        item_type         = st.selectbox("Item Type", ["Feature", "Enhancement", "Defect"],
                            help="Feature = new functionality. Enhancement = improvement. Defect = bug fix.")
        platform          = st.selectbox("Platform", ["iOS", "Android", "Platform"],
                            help="Platform = backend/API work.")

    with col_b:
        story_points = st.selectbox("Story Points", [1, 2, 3, 5, 8, 13, 21],
                        help="Team's complexity score. Higher = more complex = more hours.")
        original_est = st.number_input("Original Estimate (hrs)",
                        min_value=1.0, max_value=300.0, value=40.0, step=1.0,
                        help="How many hours the team originally estimated for this work item.")

    with col_c:
        team_size    = st.selectbox("Team Size", [3, 8, 10],
                        help="Number of people on the team.")
        feature_size = st.selectbox("Feature Size", ["S", "M", "L"],
                        help="S = Small, M = Medium, L = Large.")
        has_qa       = st.selectbox("Has QA Task?", ["Yes", "No"],
                        help="Does this story require quality assurance testing?")

    type_enc = {"Feature": 0, "Enhancement": 1, "Defect": 2}
    plat_enc = {"iOS": 0, "Android": 1, "Platform": 2}
    size_enc = {"S": 1, "M": 2, "L": 3}
    release_sprint = {
        "REL-X2": 5,  "REL-X3": 10, "REL-X4": 15,
        "REL-X5": 20, "REL-X6": 25, "REL-X7": 30,
        "REL-X8": 35, "REL-X9": 40,
        "REL-X10 (new)": 45, "REL-X11 (new)": 50,
        "Future Release": 55
    }
    sprint_num = release_sprint.get(release_sel, 55)

    input_data = {
        "Story_Points":     story_points,
        "Original_Est_Hrs": original_est,
        "Bug_Rework_Hrs":   0.0,
        "N_Tasks":          4,
        "Item_Type_enc":    type_enc[item_type],
        "Feature_Size_enc": size_enc[feature_size],
        "Platform_enc":     plat_enc[platform],
        "Has_QA_enc":       1 if has_qa == "Yes" else 0,
        "Team_Type_enc":    1,
        "Team_Size":        team_size,
        "Sprint_Number":    sprint_num,
    }

    X_input = pd.DataFrame([input_data])[FEATS]

    # Show live prediction before chatting
    xgb_pred  = xgb_m.predict(X_input)[0]
    low_pred  = ql_m.predict(X_input)[0]
    mid_pred  = qm_m.predict(X_input)[0]
    high_pred = qh_m.predict(X_input)[0]

    st.divider()
    st.subheader("Step 2 — See the instant prediction")
    st.markdown("*Updates automatically as you change the work item details above.*")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Low Estimate",    f"{low_pred:.0f} hrs",  help="Optimistic scenario")
    p2.metric("Central Estimate",f"{mid_pred:.0f} hrs",  help="Most likely outcome")
    p3.metric("High Estimate",   f"{high_pred:.0f} hrs", help="Budget for this to be safe")
    p4.metric("XGBoost Best",    f"{xgb_pred:.1f} hrs",  help="Best single estimate from the model")

    st.divider()
    st.subheader("Step 3 — Ask the Copilot Assistant")
    st.markdown("*Ask any planning question. The assistant uses historical data from all past releases to answer.*")

    import shap
    sv    = shap_exp(X_input).values[0]
    pairs = sorted(zip(FEATS, sv), key=lambda x: abs(x[1]), reverse=True)
    top3  = pairs[:3]
    driver_txt = ""
    for feat, val in top3:
        direction = "increases" if val > 0 else "decreases"
        driver_txt += (f"\n  - {feat.replace('_enc','').replace('_',' ')}: "
                       f"{direction} estimate by {abs(val):.1f} hrs")

    dists, idxs = knn_m.kneighbors(X_input, n_neighbors=3)
    analogy_txt = ""
    for rank, (d, i) in enumerate(zip(dists[0], idxs[0]), 1):
        past = train_df.iloc[i]
        overrun = ((past["Actual_Hrs"] - past["Original_Est_Hrs"])
                   / past["Original_Est_Hrs"] * 100)
        analogy_txt += (f"\n  {rank}. {past['Item_Type']} | "
                        f"{past['Story_Points']:.0f} pts | "
                        f"Est: {past['Original_Est_Hrs']:.0f} hrs | "
                        f"Actual: {past['Actual_Hrs']:.0f} hrs | "
                        f"Overrun: {overrun:+.0f}%")

    system_prompt = f"""You are a helpful AI planning assistant for Fiserv project managers.
You help estimate effort for software work items using machine learning predictions
grounded in historical delivery data from Fiserv's past releases.

IMPORTANT: The model was trained on releases REL-X2 through REL-X9. It learns
PATTERNS (how item type, story points, platform, and team size relate to actual hours)
that apply to ANY release — including new releases like REL-X10 or future releases.
When asked about new releases or new sprints, confidently use these cross-release patterns.

WORK ITEM BEING ESTIMATED:
  Release: {release_sel}
  Type: {item_type} | Platform: {platform}
  Story Points: {story_points} | Original Estimate: {original_est} hrs
  Team Size: {team_size} | Feature Size: {feature_size} | Has QA: {has_qa}

MODEL PREDICTIONS (from 5 ML models trained on 435 past stories):
  Low estimate:    {low_pred:.0f} hrs  (optimistic)
  Central:         {mid_pred:.0f} hrs  (most likely)
  High estimate:   {high_pred:.0f} hrs (budget for this)
  XGBoost:         {xgb_pred:.1f} hrs

KEY DRIVERS (what is pushing this estimate up or down):
{driver_txt}

3 MOST SIMILAR PAST STORIES (from releases REL-X2 to REL-X9):
{analogy_txt}

Answer the project manager's question clearly and conversationally.
Reference the specific numbers above. Under 150 words."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_input := st.chat_input("e.g. How long will this take? Should I be worried about overruns?"):

        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        try:
            from groq import Groq
            client = Groq(api_key="your-groq-key-here")
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": prompt_input}
                ],
                max_tokens=300
            )
            reply = response.choices[0].message.content
        except Exception as e:
            reply = f"Error connecting to AI: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
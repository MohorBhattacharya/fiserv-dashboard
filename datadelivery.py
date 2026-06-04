import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="Fiserv Effort Estimation — AI Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data(ttl=0)
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

@st.cache_resource(ttl=0)
def load_models():
    import joblib
    import pickle
    import shap
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

df, tasks, release = load_data()
xgb_m, knn_m, ql_m, qm_m, qh_m, shap_exp, train_df, FEATS = load_models()

# ── Release-level overrun summary (used by both the dashboard and the Copilot) ──
@st.cache_data(ttl=0)
def build_release_facts(_release, _df):
    rel = _release.copy()
    rel["Overrun_Pct"] = (
        (rel["Total_Actual_Hrs"] - rel["Total_Orig_Est_Hrs"])
        / rel["Total_Orig_Est_Hrs"] * 100
    ).round(1)
    lines = []
    for _, r in rel.iterrows():
        rid = r["Release_ID"]
        status = "UNDER budget" if r["Overrun_Pct"] < 0 else "OVER budget"
        # work-type breakdown for this release
        sub = _df[_df["Release_ID"] == rid]
        if len(sub):
            by_type = (sub.groupby("Item_Type")["Overrun_Pct_clean"].mean()
                       .sort_values(ascending=False))
            worst = by_type.index[0] if len(by_type) else "n/a"
            worst_pct = by_type.iloc[0] if len(by_type) else 0.0
            type_txt = (f" Worst work type: {worst} (avg {worst_pct:+.0f}% overrun). "
                        f"Stories: {len(sub)}.")
        else:
            type_txt = ""
        lines.append(
            f"  {rid}: estimated {r['Total_Orig_Est_Hrs']:.0f} hrs, "
            f"actual {r['Total_Actual_Hrs']:.0f} hrs, "
            f"{r['Overrun_Pct']:+.1f}% ({status})." + type_txt
        )
    return rel, "\n".join(lines)

release_overrun_facts, RELEASE_FACTS_TXT = build_release_facts(release, df)

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Fiserv_logo.svg/320px-Fiserv_logo.svg.png",
    width=180
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "🤖 Copilot Assistant"])

color_map = {"Defect": "#D85A30", "Enhancement": "#BA7517", "Feature": "#2E75B6"}

type_enc = {"Feature": 0, "Enhancement": 1, "Defect": 2}
plat_enc = {"iOS": 0, "Android": 1, "Platform": 2}
size_enc = {"S": 1, "M": 2, "L": 3}
release_sprint = {
    "REL-X2": 5,  "REL-X3": 10, "REL-X4": 15,
    "REL-X5": 20, "REL-X6": 25, "REL-X7": 30,
    "REL-X8": 35, "REL-X9": 40, "Future Release": 55
}

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    total_stories = len(df)
    st.title("📊 Fiserv Effort Estimation Dashboard")
    st.markdown(f"""
    > This dashboard analyses **{total_stories} completed User Stories** across 8 releases.
    > Use the filters on the left to drill into specific releases, platforms, work types, or team sizes.
    > Every chart updates automatically when you change a filter.
    """)
    st.divider()

    st.sidebar.title("Filters")
    st.sidebar.markdown("*Narrow down the data shown in all charts below.*")

    releases   = ["All"] + sorted(df["Release_ID"].unique().tolist())
    platforms  = ["All"] + sorted(df["Platform"].dropna().unique().tolist())
    itypes     = ["All"] + sorted(df["Item_Type"].unique().tolist())
    team_sizes = ["All"] + sorted(df["Team_Size"].dropna().unique().astype(int).tolist())

    sel_release  = st.sidebar.selectbox("Release",   releases)
    sel_platform = st.sidebar.selectbox("Platform",  platforms)
    sel_itype    = st.sidebar.selectbox("Item Type", itypes)
    sel_teamsize = st.sidebar.selectbox(
        "Team Size",
        team_sizes,
        help="Filter by team size to see how effort and overrun vary across different sized teams."
    )

    filtered = df.copy()
    if sel_release  != "All": filtered = filtered[filtered["Release_ID"] == sel_release]
    if sel_platform != "All": filtered = filtered[filtered["Platform"]   == sel_platform]
    if sel_itype    != "All": filtered = filtered[filtered["Item_Type"]  == sel_itype]
    if sel_teamsize != "All": filtered = filtered[filtered["Team_Size"]  == int(sel_teamsize)]

    if len(filtered) == 0:
        st.warning("⚠️ No stories match this filter combination. Try adjusting the filters.")
        st.stop()
    elif len(filtered) < 20:
        st.info(f"ℹ️ Only {len(filtered)} stories match this filter — results may not be fully representative.")

    total     = len(filtered)
    avg_hrs   = filtered["Actual_Hrs"].mean()
    med_hrs   = filtered["Actual_Hrs"].median()
    over_pct  = (filtered["Overrun_Pct_clean"] > 0).sum() / len(filtered) * 100
    defect_ov = filtered[filtered["Item_Type"] == "Defect"]["Overrun_Pct_clean"].mean()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Stories",         f"{total}",
              help="Number of completed User Stories in this filter")
    k2.metric("Avg Actual Hours",      f"{avg_hrs:.1f} hrs",
              help="Average hours a story actually took")
    k3.metric("Median Actual Hours",   f"{med_hrs:.1f} hrs",
              help="Middle value — less affected by outliers than the average")
    k4.metric("Stories Over Estimate", f"{over_pct:.0f}%",
              help="Percentage of stories that took longer than originally estimated")
    k5.metric("Defect Avg Overrun",    f"+{defect_ov:.0f}%" if not np.isnan(defect_ov) else "N/A",
              help="How much longer Defects took vs their estimate on average")
    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📉 Overrun Analysis",
        "📊 Release Overrun",
        "⏱️ Effort Patterns",
        "🔗 What Predicts Hours?",
        "📋 Raw Data",
        "🚨 Spillover Risk"
    ])

    with tab1:
        st.markdown("### Why do projects run over time?")
        st.markdown("""
        These charts show the **estimation accuracy problem** — how often and by how much
        work takes longer than planned.
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
                               yaxis_title="Average Overrun %", xaxis_title="Work Type")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("#### Overrun % by Team Size")
            st.markdown("*Does team size affect how much work overruns? Larger teams may handle complexity differently.*")
            team_overrun = (
                filtered.groupby("Team_Size")["Overrun_Pct_clean"]
                .mean().reset_index()
                .rename(columns={"Overrun_Pct_clean": "Avg Overrun %", "Team_Size": "Team Size"})
                .sort_values("Team Size")
            )
            team_overrun["Team Size"] = team_overrun["Team Size"].astype(str) + " people"
            fig_team = px.bar(team_overrun, x="Team Size", y="Avg Overrun %",
                              color="Avg Overrun %",
                              color_continuous_scale="Reds",
                              text="Avg Overrun %")
            fig_team.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_team.update_layout(showlegend=False, height=380,
                                   coloraxis_showscale=False,
                                   yaxis_title="Average Overrun %")
            st.plotly_chart(fig_team, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### How Much Do Individual Stories Overrun?")
            st.markdown("*The red line is zero — no overrun. Bars to the left came in under budget. Bars to the right overran.*")
            fig8 = px.histogram(filtered, x="Overrun_Pct_clean", nbins=40,
                                color_discrete_sequence=["#7F77DD"],
                                labels={"Overrun_Pct_clean": "Overrun % (positive = took longer, negative = under budget)"})
            fig8.add_vline(x=0, line_dash="dash", line_color="red",
                           annotation_text="← Under budget  |  Over estimate →",
                           annotation_position="top left")
            fig8.update_layout(height=360)
            st.plotly_chart(fig8, use_container_width=True)

        with col4:
            st.markdown("#### Average Hours by Platform")
            st.markdown("*Does iOS, Android, or backend work take different amounts of time?*")
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
            fig4.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("#### Breakdown by Work Type")
        pie_col, _ = st.columns([1, 1])
        with pie_col:
            type_counts = filtered["Item_Type"].value_counts().reset_index()
            type_counts.columns = ["Item Type", "Count"]
            fig9 = px.pie(type_counts, names="Item Type", values="Count",
                          color="Item Type", color_discrete_map=color_map,
                          title="What proportion is Features vs Enhancements vs Defects?",
                          hole=0.4)
            fig9.update_layout(height=340)
            st.plotly_chart(fig9, use_container_width=True)

    with tab2:
        st.markdown("### Which releases had the biggest overruns?")
        st.markdown("""
        This tab shows **release-level overrun** — comparing total estimated hours against
        what was actually delivered, release by release.
        Green bars indicate the release came in **under budget**; red bars indicate **over budget**.
        """)

        st.markdown("#### Overrun % per Release")
        st.markdown("*Positive = over budget. Negative = under budget.*")

        release_overrun = release.copy()
        release_overrun["Overrun_Pct"] = (
            (release_overrun["Total_Actual_Hrs"] - release_overrun["Total_Orig_Est_Hrs"])
            / release_overrun["Total_Orig_Est_Hrs"] * 100
        ).round(1)

        fig_rov = px.bar(
            release_overrun,
            x="Release_ID",
            y="Overrun_Pct",
            text="Overrun_Pct",
            color="Overrun_Pct",
            color_continuous_scale=["#1D9E75", "#FFDD57", "#D85A30"],
            range_color=[-20, 30],
            labels={"Release_ID": "Release", "Overrun_Pct": "Overrun %"}
        )
        fig_rov.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_rov.add_hline(y=0, line_dash="dash", line_color="white",
                          annotation_text="Zero overrun line", annotation_position="top left")
        fig_rov.update_layout(height=420, coloraxis_showscale=False,
                              xaxis_title="Release", yaxis_title="Overrun %")
        st.plotly_chart(fig_rov, use_container_width=True)

        st.markdown("#### Estimated vs Actual Hours — Release by Release")
        st.markdown("*Light blue = original estimate. Orange = actual hours spent. When orange is taller than blue, the release went over budget.*")
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Original Estimate",
            x=release["Release_ID"], y=release["Total_Orig_Est_Hrs"],
            marker_color="#B5D4F4"
        ))
        fig5.add_trace(go.Bar(
            name="Actual Hours",
            x=release["Release_ID"], y=release["Total_Actual_Hrs"],
            marker_color="#D85A30"
        ))
        fig5.update_layout(
            barmode="group",
            xaxis_title="Release", yaxis_title="Total Hours", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Release Summary Table")
        summary_display = release_overrun[["Release_ID", "Total_Orig_Est_Hrs", "Total_Actual_Hrs", "Overrun_Pct"]].copy()
        summary_display.columns = ["Release", "Estimated Hours", "Actual Hours", "Overrun %"]
        summary_display["Overrun %"] = summary_display["Overrun %"].apply(
            lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
        )
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        # FIX — tooltip rewritten: Copilot answers release questions directly, no form to fill
        st.info("""💡 **How to use this tab:** Spot a release with a high overrun % in the chart above.
Use the **Release filter** on the left to drill into that release's individual stories across the other tabs (Overrun Analysis, Effort Patterns, Raw Data).
Want the AI to explain it? Open the **🤖 Copilot Assistant** page and simply type your question — for example *"Why did REL-X6 overrun?"* You do **not** need to fill in any form first; the Copilot already knows every release's figures.""")

    with tab3:
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
            st.markdown("*Higher complexity consistently means more hours. Defects (orange) sit above the trend line — they take longer than their complexity suggests.*")
            fig3 = px.scatter(filtered, x="Story_Points", y="Actual_Hrs",
                              color="Item_Type", color_discrete_map=color_map,
                              hover_data=["Epic", "Feature_Size", "Team_ID", "Original_Est_Hrs"],
                              labels={"Story_Points": "Story Points (Complexity Score)",
                                      "Actual_Hrs": "Actual Hours Taken",
                                      "Item_Type": "Work Type"},
                              render_mode="svg")
            x = filtered["Story_Points"].values
            y = filtered["Actual_Hrs"].values
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
                y_line = p(x_line)
                fig3.add_scatter(x=x_line, y=y_line,
                                 mode="lines",
                                 line=dict(color="white", width=2),
                                 name="Trend",
                                 showlegend=False)
            fig3.update_layout(height=380)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Distribution of Actual Hours by Team Size")
        st.markdown("*Select a Team Size filter on the left to see how hours distribute for that team specifically.*")
        fig_dist_team = px.histogram(
            filtered, x="Actual_Hrs", nbins=30,
            facet_col="Team_Size",
            color_discrete_sequence=["#2E75B6"],
            labels={"Actual_Hrs": "Actual Hours", "Team_Size": "Team Size"},
            title="Actual Hours Distribution — split by Team Size"
        )
        fig_dist_team.update_layout(height=360)
        st.plotly_chart(fig_dist_team, use_container_width=True)

    with tab4:
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
            st.markdown("*Original estimate and story points are the strongest signals.*")
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
            st.markdown("*Darker blue = stronger relationship. The bottom row shows what predicts Actual Hours.*")
            corr = filtered[numeric_cols].corr().round(2)
            fig6 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", aspect="auto")
            fig6.update_layout(height=380)
            st.plotly_chart(fig6, use_container_width=True)

    with tab5:
        st.markdown("### Underlying Data")
        st.markdown("*Every row is one completed User Story. Use the filters on the left to narrow down.*")
        display_cols = ["Story_ID", "Release_ID", "Epic", "Item_Type", "Platform",
                        "Team_Size", "Story_Points", "Original_Est_Hrs", "Actual_Hrs",
                        "Overrun_Pct_clean", "Status"]
        st.dataframe(
            filtered[display_cols].rename(columns={"Overrun_Pct_clean": "Overrun %"}),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Showing {len(filtered)} stories  |  Built by Mohor — University of Auckland Industry Placement | Fiserv Professional Services | 2026")

    with tab6:
        st.markdown("### 🚨 Will This Change Request Cause Spillover?")
        st.markdown("""
        Enter the sprint capacity details and the change request size below.
        The model predicts whether adding this work **mid-sprint** will push the team into overrun.
        """)

        st.divider()
        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### Sprint Capacity")
            sp_team_size   = st.slider("Team Size (people)", 1, 10, 5, key="sp_ts",
                                        help="Any team size from 1 to 10 people.")
            sp_hrs_per_day = st.number_input("Working hrs per person per day",
                                              min_value=4.0, max_value=10.0, value=7.0, step=0.5,
                                              help="Typical = 6–8 hrs.")
            sp_days        = st.number_input("Sprint length (days)",
                                              min_value=5, max_value=30, value=10, step=1)
            total_capacity = sp_team_size * sp_hrs_per_day * sp_days
            st.metric("Total Sprint Capacity", f"{total_capacity:.0f} hrs")

        with col_s2:
            st.markdown("#### Current Sprint State")
            is_mid_sprint = st.checkbox("⚡ This is a MID-SPRINT change request", value=True)
            pct_complete  = st.slider("Sprint % complete", 0, 100, 50,
                                      disabled=not is_mid_sprint,
                                      help="Each working day = 10% of a 10-day sprint. e.g. 3 days in = 30%.")

            # Remaining capacity = Total capacity × (1 − sprint % complete).
            if is_mid_sprint:
                remaining_capacity = total_capacity * (1 - pct_complete / 100)
            else:
                remaining_capacity = total_capacity

            remaining_capacity = max(remaining_capacity, 0)
            st.metric("Remaining Capacity", f"{remaining_capacity:.0f} hrs",
                      help="Total sprint capacity × (1 − sprint % complete).")

        st.divider()
        st.markdown("#### Change Request Details")
        cr_col1, cr_col2, cr_col3 = st.columns(3)

        with cr_col1:
            cr_story_pts = st.slider("Story Points", 1, 21, 5, key="cr_sp")
            cr_orig_est  = st.number_input("Original Estimate (hrs)",
                                            min_value=1.0, max_value=300.0,
                                            value=float(cr_story_pts * 8),
                                            step=1.0, key="cr_oe")
        with cr_col2:
            cr_item_type = st.selectbox("Item Type", ["Feature", "Enhancement", "Defect"], key="cr_it")
            cr_platform  = st.selectbox("Channel or Domain", ["iOS", "Android", "Platform"], key="cr_pl")
        with cr_col3:
            cr_feat_size = st.selectbox("Change Size", ["S", "M", "L"], key="cr_fs")
            cr_has_qa    = st.selectbox("Has QA?", ["Yes", "No"], key="cr_qa")

        cr_sprint_num = release_sprint.get(sel_release, 30) if sel_release != "All" else 30

        cr_input = pd.DataFrame([{
            "Story_Points":     cr_story_pts,
            "Original_Est_Hrs": cr_orig_est,
            "Bug_Rework_Hrs":   0.0,
            "N_Tasks":          4,
            "Item_Type_enc":    type_enc[cr_item_type],
            "Feature_Size_enc": size_enc[cr_feat_size],
            "Platform_enc":     plat_enc[cr_platform],
            "Has_QA_enc":       1 if cr_has_qa == "Yes" else 0,
            "Team_Type_enc":    1,
            "Team_Size":        sp_team_size,
            "Sprint_Number":    cr_sprint_num,
        }])[FEATS]

        cr_xgb  = float(xgb_m.predict(cr_input)[0])
        # Sort the three quantile outputs so Low <= Central <= High always holds
        _crq = sorted([float(ql_m.predict(cr_input)[0]),
                       float(qm_m.predict(cr_input)[0]),
                       float(qh_m.predict(cr_input)[0])])
        cr_low, cr_mid, cr_high = _crq[0], _crq[1], _crq[2]

        st.divider()
        st.markdown("#### Model Predictions for This Change Request")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Low Estimate",      f"{cr_low:.0f} hrs",  help="Optimistic scenario")
        m2.metric("Central Estimate",  f"{cr_mid:.0f} hrs",  help="Most likely")
        m3.metric("High Estimate",     f"{cr_high:.0f} hrs", help="Worst case")
        m4.metric("Primary Model",     f"{cr_xgb:.0f} hrs",  help="Best single estimate (Random Forest)")

        st.divider()
        st.markdown("#### 🔎 Spillover Verdict")

        headroom    = remaining_capacity - cr_xgb
        headroom_lo = remaining_capacity - cr_low
        headroom_hi = remaining_capacity - cr_high

        if remaining_capacity == 0:
            risk_level = "🔴 HIGH RISK — No capacity remaining"
            risk_color = "#D85A30"
        elif headroom > 0 and headroom_lo > 0 and headroom_hi > 0:
            risk_level = "🟢 LOW RISK — Fits comfortably within remaining capacity"
            risk_color = "#1D9E75"
        elif headroom > 0 and headroom_hi > 0:
            risk_level = "🟡 MEDIUM RISK — Fits on central estimate, high scenario is tight"
            risk_color = "#FFDD57"
        elif headroom > 0 and headroom_hi <= 0:
            risk_level = "🟠 MEDIUM-HIGH RISK — Central fits but worst case causes spillover"
            risk_color = "#E8882A"
        else:
            risk_level = "🔴 HIGH RISK — Spillover certain across all scenarios"
            risk_color = "#D85A30"

        st.markdown(f"""
        <div style="background:{risk_color}22; border-left: 5px solid {risk_color};
                    padding: 16px; border-radius: 8px; margin-bottom: 16px;">
            <h3 style="color:{risk_color}; margin:0;">{risk_level}</h3>
        </div>
        """, unsafe_allow_html=True)

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Remaining Capacity",  f"{remaining_capacity:.0f} hrs")
        v2.metric("Predicted (Primary Model)", f"{cr_xgb:.0f} hrs")
        v3.metric("Headroom (Central)",  f"{headroom:+.0f} hrs",
                  delta_color="normal" if headroom >= 0 else "inverse")
        v4.metric("Worst-Case Headroom", f"{headroom_hi:+.0f} hrs",
                  delta_color="normal" if headroom_hi >= 0 else "inverse")

        bar_data = pd.DataFrame({
            "Scenario": ["Remaining Capacity", "Low Estimate", "Central Estimate", "High Estimate"],
            "Hours":    [remaining_capacity, cr_low, cr_xgb, cr_high],
        })
        fig_sp = px.bar(bar_data, x="Scenario", y="Hours",
                        color="Scenario",
                        color_discrete_map={
                            "Remaining Capacity": "#2E75B6",
                            "Low Estimate":       "#1D9E75",
                            "Central Estimate":   "#FFDD57",
                            "High Estimate":      "#D85A30"
                        },
                        text="Hours")
        fig_sp.update_traces(texttemplate="%{text:.0f} hrs", textposition="outside")
        fig_sp.add_hline(y=remaining_capacity, line_dash="dash", line_color="white",
                         annotation_text="Capacity ceiling", annotation_position="top right")
        fig_sp.update_layout(height=400, showlegend=False,
                             yaxis_title="Hours", xaxis_title="")
        st.plotly_chart(fig_sp, use_container_width=True)

        st.divider()
        st.markdown("#### 📋 How Was This Verdict Reached?")
        narrative_remaining = f"remaining capacity = {total_capacity:.0f} × (1 − {pct_complete}%) = **{remaining_capacity:.0f} hrs**"
        narrative_xgb       = f"XGBoost predicts this change request will take **{cr_xgb:.0f} hrs**"
        narrative_headroom  = f"Headroom = {remaining_capacity:.0f} − {cr_xgb:.0f} = **{headroom:+.0f} hrs**"

        st.markdown(f"""
        Here is a step-by-step explanation of how the system arrived at the **{risk_level}** verdict:

        **Step 1 — Sprint Capacity**
        The team has **{sp_team_size} people**, working **{sp_hrs_per_day:.0f} hrs/day** over a **{sp_days:.0f}-day sprint**.
        Total sprint capacity = {sp_team_size} × {sp_hrs_per_day:.0f} × {sp_days:.0f} = **{total_capacity:.0f} hrs**.

        **Step 2 — Remaining Capacity**
        The sprint is **{pct_complete}% complete**, so the {narrative_remaining}.

        **Step 3 — Effort Prediction**
        For a **{cr_item_type}** with **{cr_story_pts} story points** and an original estimate of **{cr_orig_est:.0f} hrs**,
        the {narrative_xgb}. The quantile model also gives a range: Low {cr_low:.0f} hrs → High {cr_high:.0f} hrs.

        **Step 4 — Verdict**
        {narrative_headroom}.
        {"All three estimates (low, central, high) fit within remaining capacity — low spillover risk." if headroom > 0 and headroom_lo > 0 and headroom_hi > 0
          else "The central estimate fits but the worst-case scenario would exceed capacity." if headroom > 0 and headroom_hi <= 0
          else "The predicted hours exceed remaining capacity — spillover is likely."}
        """)

        st.info("💡 Fill in your actual sprint state above. The verdict and this narrative update automatically.")

        st.divider()
        st.markdown("#### 🔗 What Drives the Spillover Verdict?")
        st.markdown("""
        The verdict is determined by two independent components. Understanding which inputs affect which component helps explain why the verdict changes the way it does.
        """)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### Component 1 — Remaining Capacity")
            st.markdown("*Pure arithmetic — no ML involved. Remaining capacity = Total Capacity × (1 − Sprint % Complete):*")
            capacity_data = pd.DataFrame({
                "Input": ["Sprint % Complete", "Team Size", "Sprint Length (days)", "Working Hrs/Day"],
                "Impact": ["🔴 High", "🟠 Medium", "🟠 Medium", "🟠 Medium"],
                "How": [
                    "More % complete = less remaining capacity",
                    "Larger team = more total capacity",
                    "Longer sprint = more total capacity",
                    "More hrs/day = more total capacity"
                ]
            })
            st.dataframe(capacity_data, use_container_width=True, hide_index=True)

        with col_d2:
            st.markdown("##### Component 2 — Predicted Effort (Primary Model)")
            st.markdown("*These inputs feed the ML model. Story Points and Original Estimate are the strongest signals:*")
            effort_data = pd.DataFrame({
                "Input": ["Story Points", "Original Estimate (hrs)", "Item Type", "Change Size", "Has QA", "Channel or Domain"],
                "Impact": ["🔴 High", "🔴 High", "🟡 Low", "🟡 Low", "🟡 Low", "🟡 Low"],
                "How": [
                    "Higher points = more predicted hours",
                    "Higher estimate = more predicted hours",
                    "Defects tend to overrun most",
                    "Larger size = slightly more hours",
                    "QA adds effort",
                    "Minor effect on prediction"
                ]
            })
            st.dataframe(effort_data, use_container_width=True, hide_index=True)

        st.markdown("""
        **Verdict logic:** The system compares remaining capacity against the Primary Model prediction.
        - 🟢 **Low Risk** — all three estimates (low, central, high) fit within remaining capacity
        - 🟡 **Medium Risk** — central fits but high scenario is tight
        - 🟠 **Medium-High** — central fits but worst case causes spillover
        - 🔴 **High Risk** — predicted hours exceed remaining capacity across all scenarios
        """)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — COPILOT ASSISTANT  (redesigned: chat-first, form optional)
# ═════════════════════════════════════════════════════════════
elif page == "🤖 Copilot Assistant":

    st.title("🤖 Copilot Assistant")

    # Clear intro explaining the TWO things you can do here
    st.markdown("""
    Ask me anything about Fiserv delivery — **just type a question below to start.**
    """)
    c_intro1, c_intro2 = st.columns(2)
    with c_intro1:
        st.info("""**💬 Ask about releases & past data**

No setup needed. Try:
- *“Why did REL-X6 overrun?”*
- *“Which release performed best?”*
- *“What type of work overruns most?”*""")
    with c_intro2:
        st.success("""**📐 Estimate a new work item**

Open **“Describe a work item”** below to set the details, then ask:
- *“How long will this take?”*
- *“Will it cause spillover mid-sprint?”*""")

    st.divider()

    # ── OPTIONAL work-item form, collapsed by default so the page leads with chat ──
    with st.expander("📐 Describe a work item  (optional — only needed for effort estimates)", expanded=False):
        st.caption("Set these details to get a model-based estimate for a *specific* new work item. "
                   "For questions about past releases you can skip this entirely.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            existing_releases = sorted(df["Release_ID"].unique().tolist())
            release_options   = existing_releases + ["Future Release"]
            release_sel       = st.selectbox("Release", release_options,
                                help="Select an existing release or Future Release for upcoming ones.")
            item_type         = st.selectbox("Item Type", ["Feature", "Enhancement", "Defect"],
                                help="Feature = new functionality. Enhancement = improvement. Defect = bug fix.")
            platform          = st.selectbox("Channel or Domain", ["iOS", "Android", "Platform"],
                                help="iOS / Android = mobile channels. Platform = backend/API work.")
        with col_b:
            story_points = st.slider("Story Points", 1, 21, 5,
                            help="Team's complexity score. Higher = more complex = more hours.")
            original_est = st.number_input("Original Estimate (hrs)",
                            min_value=1.0, max_value=300.0, value=40.0, step=1.0,
                            help="How many hours the team originally estimated for this work item.")
        with col_c:
            team_size    = st.slider("Team Size", 1, 10, 5, key="cop_team_size",
                            help="Number of people on the team.")
            feature_size = st.selectbox("Change Size", ["S", "M", "L"],
                            help="S = Small, M = Medium, L = Large.")
            has_qa       = st.selectbox("Has QA Task?", ["Yes", "No"],
                            help="Does this story require quality assurance testing?")

        st.markdown("###### Mid-Sprint Context (optional)")
        st.caption("Fill this in if the customer is requesting a change while a sprint is already in progress.")
        mid_col1, mid_col2 = st.columns(2)
        with mid_col1:
            mid_sprint_cr = st.checkbox("⚡ This is a mid-sprint change request", value=False)
            mid_pct       = st.slider("Sprint % complete", 0, 100, 50,
                            disabled=not mid_sprint_cr, key="cop_mid",
                            help="Each working day = 10% of a 10-day sprint. e.g. 3 days in = 30%.")
        with mid_col2:
            default_cap  = float(team_size * 7 * 14)
            mid_capacity = st.number_input("Total sprint capacity (hrs)",
                            min_value=10.0, value=default_cap, step=10.0,
                            disabled=not mid_sprint_cr, key="cop_cap",
                            help="Team Size × hrs/day × sprint days. Auto-filled as 7 hrs/day × 14 days.")

        # live predictions for the described item
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
        xgb_pred  = float(xgb_m.predict(X_input)[0])
        # Sort the three quantile outputs so Low <= Central <= High always holds
        _q = sorted([float(ql_m.predict(X_input)[0]),
                     float(qm_m.predict(X_input)[0]),
                     float(qh_m.predict(X_input)[0])])
        low_pred, mid_pred, high_pred = _q[0], _q[1], _q[2]

        if mid_sprint_cr:
            mid_remaining = mid_capacity * (1 - mid_pct / 100)
            mid_fits      = mid_remaining >= xgb_pred
            mid_headroom  = mid_remaining - xgb_pred
            verdict_label = "✅ FITS in remaining capacity" if mid_fits else "🚨 SPILLOVER RISK"
            st.metric("Remaining Sprint Capacity", f"{mid_remaining:.0f} hrs",
                      delta=f"{verdict_label}  |  Headroom: {mid_headroom:+.0f} hrs",
                      delta_color="normal" if mid_fits else "inverse")

        st.markdown("**Instant prediction for this work item:**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Low",     f"{low_pred:.0f} hrs")
        p2.metric("Central", f"{mid_pred:.0f} hrs")
        p3.metric("High",    f"{high_pred:.0f} hrs")
        p4.metric("Primary Model", f"{xgb_pred:.1f} hrs")
        st.caption("These numbers are passed to the Copilot, so you can ask follow-up questions about this item below.")

    # ── Build the SHAP drivers + nearest-neighbour analogies for the described item ──
    import shap as shap_lib
    sv    = shap_exp(X_input).values[0]
    pairs = sorted(zip(FEATS, sv), key=lambda x: abs(x[1]), reverse=True)
    top3  = pairs[:3]
    driver_txt = ""
    for feat, val in top3:
        direction = "above" if val > 0 else "below"
        driver_txt += (f"\n  - {feat.replace('_enc','').replace('_',' ')}: "
                       f"pushes this estimate {abs(val):.1f} hrs {direction} the average story")

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

    mid_sprint_context = ""
    if mid_sprint_cr:
        mid_remaining = mid_capacity * (1 - mid_pct / 100)
        fits          = mid_remaining >= xgb_pred
        mid_sprint_context = f"""
MID-SPRINT CONTEXT (for the described work item):
  Sprint is {mid_pct}% complete.
  Total sprint capacity: {mid_capacity:.0f} hrs
  Remaining capacity: {mid_remaining:.0f} hrs (= {mid_capacity:.0f} × (1 − {mid_pct}%))
  Primary model predicts this item needs: {xgb_pred:.1f} hrs
  Verdict: {"FITS within remaining capacity." if fits else "SPILLOVER RISK — predicted hours exceed remaining capacity."}
  Headroom: {mid_remaining - xgb_pred:+.0f} hrs
"""

    # ── System prompt now ALSO includes release-level facts so the Copilot can
    #    answer "why did REL-X6 overrun?" with no form filled in. ──
    system_prompt = f"""You are a helpful AI planning assistant for Fiserv project managers.
You answer two kinds of questions:
  (1) Questions about PAST RELEASES and historical delivery data — answer these directly from the RELEASE DATA below. No work-item form is needed.
  (2) Effort ESTIMATES for a specific new work item — use the WORK ITEM and MODEL PREDICTIONS below.

Figure out which kind of question the user is asking and answer accordingly. If they ask about a release (e.g. "why did REL-X6 overrun?"), use the RELEASE DATA — do not ask them to fill in a form.

RELEASE DATA (all completed releases, REL-X2 through REL-X9):
{RELEASE_FACTS_TXT}

Overall: across {len(df)} completed stories, about {(df['Overrun_Pct_clean']>0).mean()*100:.0f}% overran their original estimate. Defects overrun the most; features are the most predictable.

WORK ITEM (only relevant for estimate questions; defaults if the user hasn't set the form):
  Release: {release_sel}
  Type: {item_type} | Channel or Domain: {platform}
  Story Points: {story_points} | Original Estimate: {original_est} hrs
  Team Size: {team_size} | Change Size: {feature_size} | Has QA: {has_qa}

MODEL PREDICTIONS for that work item (THIS is the answer to "how long will it take"):
  Most likely (central) estimate: {mid_pred:.0f} hrs
  Likely range: {low_pred:.0f} hrs (optimistic) to {high_pred:.0f} hrs (conservative)
  Primary Model (Random Forest): {xgb_pred:.1f} hrs

KEY DRIVERS — these explain why this estimate differs from a typical story.
They are RELATIVE effects versus the average story, NOT the estimate itself. Do not use them to calculate a new total:{driver_txt}

3 MOST SIMILAR PAST STORIES (for that work item):{analogy_txt}
{mid_sprint_context}
ANSWERING RULES:
- For "how long will this take", lead with the central estimate of {mid_pred:.0f} hrs and give the {low_pred:.0f}-{high_pred:.0f} hrs range. You may mention the XGBoost figure as a cross-check.
- NEVER invent a number outside the {low_pred:.0f}-{high_pred:.0f} hrs range. NEVER say a driver "decreases the estimate to" some total — drivers are relative only.
- Be clear and conversational, reference the specific numbers above, keep it under 150 words."""

    st.subheader("💬 Ask the Copilot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt_input := st.chat_input("e.g. Why did REL-X6 overrun?  ·  How long will this work item take?"):
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # resolve the API key from secrets or env
        api_key = None

        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            reply = ("⚠️ The AI assistant isn't configured yet. No GROQ_API_KEY was found. "
                     "Add a valid key under the app's Settings → Secrets (Streamlit Cloud) "
                     "or as an environment variable, then ask again.")
        else:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
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
                msg = str(e)
                if "invalid_api_key" in msg or "401" in msg:
                    reply = ("⚠️ The Groq API key was rejected (invalid or expired). "
                             "Generate a new key at console.groq.com/keys and update it in "
                             "Settings → Secrets as GROQ_API_KEY, then try again.")
                elif "model" in msg.lower() and ("decommission" in msg.lower() or "not found" in msg.lower()):
                    reply = ("⚠️ The configured AI model is unavailable. The model name may have "
                             "changed on Groq's side — check console.groq.com/docs/models for the "
                             "current model string.")
                else:
                    reply = f"Error connecting to AI: {msg}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
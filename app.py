"""
Pumzika Hackathon — Amenities Impact Optimization
Streamlit App
Model: loaded from pumzika_model.pkl (do NOT retrain)
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pumzika · Amenity Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary":   "#C0392B",
    "secondary": "#2C3E50",
    "accent":    "#E67E22",
    "light":     "#ECF0F1",
    "muted":     "#95A5A6",
    "bg":        "#FAFAF8",
    "card":      "#FFFFFF",
}

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {PALETTE['bg']};
  }}
  h1, h2, h3 {{
    font-family: 'Playfair Display', serif;
    color: {PALETTE['secondary']};
  }}
  .metric-card {{
    background: {PALETTE['card']};
    border-left: 4px solid {PALETTE['primary']};
    border-radius: 6px;
    padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 8px;
  }}
  .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {PALETTE['primary']};
    font-family: 'Playfair Display', serif;
  }}
  .metric-label {{
    font-size: 0.8rem;
    color: {PALETTE['muted']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .section-rule {{
    border: none;
    border-top: 2px solid {PALETTE['light']};
    margin: 28px 0 20px 0;
  }}
  .rec-card {{
    background: {PALETTE['card']};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    border-top: 3px solid var(--priority-color, {PALETTE['accent']});
  }}
  .badge-high   {{ color: {PALETTE['primary']}; font-weight: 600; }}
  .badge-medium {{ color: {PALETTE['accent']};  font-weight: 600; }}
  .badge-low    {{ color: {PALETTE['muted']};   font-weight: 500; }}
  .stSidebar {{ background: {PALETTE['secondary']}; }}
  .stSidebar * {{ color: {PALETTE['light']} !important; }}
  div[data-testid="stSidebarContent"] {{
    background: {PALETTE['secondary']} !important;
  }}
  .warning-box {{
    background: #FFF3CD;
    border-left: 4px solid #F39C12;
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 0.88rem;
    color: #7D6608;
    margin: 10px 0;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODEL (once, cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("pumzika_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts = load_model()
except FileNotFoundError:
    st.error("❌ `pumzika_model.pkl` not found. Place it in the same folder as `app.py`.")
    st.stop()

model          = artifacts["model"]
feature_names  = artifacts["feature_names"]
feat_imp       = artifacts["feat_imp"]
kept_amenities = artifacts["kept_amenities"]
adoption_rates = artifacts["adoption_rates"]
holdout_rmse   = artifacts["holdout_rmse"]
holdout_r2     = artifacts["holdout_r2"]
y_dev_mean     = artifacts["y_dev_mean"]
y_dev_std      = artifacts["y_dev_std"]

STRUCTURAL = {
    "accommodates","bedrooms","beds","bathrooms",
    "property_type","room_type","minimum_nights",
    "maximum_nights","instant_bookable"
}
amenity_imp = feat_imp[~feat_imp["feature"].isin(STRUCTURAL)].copy().reset_index(drop=True)
max_imp     = amenity_imp["importance"].max()


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Pumzika")
    st.markdown("**Amenity Intelligence**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Importance Analysis", "💡 Recommendation Engine", "🔮 Score Predictor"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** Random Forest (300 trees)")
    st.markdown(f"**Holdout RMSE:** `{holdout_rmse:.4f}`")
    st.markdown(f"**Holdout R²:** `{holdout_r2:.4f}`")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;opacity:0.7'>Model is loaded read-only.<br>No retraining occurs.</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# HELPER: metric card
# ─────────────────────────────────────────────────────────────
def metric_card(label, value, col):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("# Amenities Impact Optimization")
    st.markdown("##### How hospitality amenities drive guest review scores in Seattle's short-term rental market")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metric_card("Listings Analysed", "3,818", c1)
    metric_card("Amenities Tracked", str(len(kept_amenities)), c2)
    metric_card("Holdout RMSE", f"{holdout_rmse:.3f} pts", c3)
    metric_card("Target Mean Score", f"{y_dev_mean:.1f} / 100", c4)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Top 15 Amenities by Impact")
        top15 = amenity_imp.head(15).sort_values("importance")
        colors = [
            PALETTE["primary"] if i >= 10 else
            PALETTE["accent"]  if i >= 5  else
            PALETTE["muted"]
            for i in range(len(top15))
        ]
        fig = go.Figure(go.Bar(
            x=top15["importance"],
            y=top15["feature"].str.replace("_", " ").str.title(),
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in top15["importance"]],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            margin=dict(l=10, r=60, t=10, b=10),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#EEE", title="RF Feature Importance"),
            yaxis=dict(tickfont=dict(size=11)),
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### Amenity Adoption Rates")
        top10_a = amenity_imp.head(10)
        adop    = [adoption_rates.get(row["feature"], 0) * 100 for _, row in top10_a.iterrows()]
        fig2 = go.Figure(go.Bar(
            x=top10_a["feature"].str.replace(" ", "<br>").str.title(),
            y=adop,
            marker_color=PALETTE["secondary"],
            text=[f"{v:.0f}%" for v in adop],
            textposition="outside",
        ))
        fig2.update_layout(
            margin=dict(l=10, r=10, t=10, b=20),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="% of Listings with Amenity", range=[0, 110], showgrid=True, gridcolor="#EEE"),
            xaxis=dict(tickfont=dict(size=9)),
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Priority Matrix — Impact vs Adoption")
    st.caption("High-impact + low-adoption amenities represent the biggest opportunity for hosts.")

    all_adop = np.array([adoption_rates.get(row["feature"], 0) * 100 for _, row in amenity_imp.iterrows()])
    fig3 = go.Figure()
    for priority, color, mask in [
        ("High Impact",   PALETTE["primary"], amenity_imp["importance"] >= amenity_imp["importance"].quantile(0.75)),
        ("Medium Impact", PALETTE["accent"],  (amenity_imp["importance"] < amenity_imp["importance"].quantile(0.75)) &
                                               (amenity_imp["importance"] >= amenity_imp["importance"].quantile(0.40))),
        ("Lower Impact",  PALETTE["muted"],   amenity_imp["importance"] < amenity_imp["importance"].quantile(0.40)),
    ]:
        idx = amenity_imp[mask].index
        fig3.add_trace(go.Scatter(
            x=all_adop[idx],
            y=amenity_imp.loc[idx, "importance"],
            mode="markers+text",
            name=priority,
            marker=dict(color=color, size=12, opacity=0.85),
            text=amenity_imp.loc[idx, "feature"].str.title(),
            textposition="top center",
            textfont=dict(size=9),
        ))
    fig3.add_vline(x=50, line_dash="dash", line_color=PALETTE["muted"], annotation_text="50% adoption")
    fig3.update_layout(
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Adoption Rate (%)", showgrid=True, gridcolor="#EEE"),
        yaxis=dict(title="Feature Importance", showgrid=True, gridcolor="#EEE"),
        font=dict(family="IBM Plex Sans"),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=80),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <strong>Interpretation note:</strong> Feature importance measures statistical association with review scores,
    not causal effect. Amenities correlated with higher-quality properties (e.g. fire extinguisher, carbon monoxide detector)
    may reflect host professionalism rather than direct guest satisfaction impact.
    Treat all findings as <em>associated with</em> higher ratings, not <em>causing</em> them.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2 — IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "🔍 Importance Analysis":
    st.markdown("# Feature Importance Analysis")
    st.markdown("##### Random Forest permutation-based importance — all features ranked")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    view = st.radio("Show features:", ["Amenities only", "All features (including structural)"], horizontal=True)
    n_show = st.slider("Number of features to display", 5, 33, 20)

    if view == "Amenities only":
        display_df = amenity_imp.head(n_show).copy()
        title_txt  = f"Top {n_show} Amenity Features by RF Importance"
    else:
        display_df = feat_imp.head(n_show).copy()
        title_txt  = f"Top {n_show} All Features by RF Importance"

    display_df = display_df.sort_values("importance")
    bar_colors = [
        PALETTE["primary"] if v >= display_df["importance"].quantile(0.75) else
        PALETTE["accent"]  if v >= display_df["importance"].quantile(0.40) else
        PALETTE["muted"]
        for v in display_df["importance"]
    ]

    fig = go.Figure(go.Bar(
        x=display_df["importance"],
        y=display_df["feature"].str.replace(" ", " ").str.title(),
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.5f}" for v in display_df["importance"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=dict(text=title_txt, font=dict(family="Playfair Display", size=16)),
        height=max(350, n_show * 22),
        margin=dict(l=10, r=80, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#EEE", title="Importance Score"),
        yaxis=dict(tickfont=dict(size=11)),
        font=dict(family="IBM Plex Sans"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Importance vs Adoption Gap")
    st.caption("Features in the top-right are impactful and already widely adopted. "
               "Top-left = high impact, low adoption = opportunity.")

    show_df = amenity_imp.copy()
    adop_arr = np.array([adoption_rates.get(r["feature"], 0) * 100 for _, r in show_df.iterrows()])
    gap = show_df["importance"] - (adop_arr / 100 * show_df["importance"].max())

    fig2 = px.scatter(
        x=adop_arr,
        y=show_df["importance"],
        text=show_df["feature"].str.title(),
        color=show_df["importance"],
        color_continuous_scale=[[0, PALETTE["muted"]], [0.5, PALETTE["accent"]], [1, PALETTE["primary"]]],
        labels={"x": "Adoption Rate (%)", "y": "Feature Importance", "color": "Importance"},
        size=show_df["importance"] * 2000 + 8,
        size_max=28,
    )
    fig2.update_traces(textposition="top center", textfont=dict(size=9))
    fig2.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans"),
        coloraxis_showscale=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Full Importance Table")
    table_df = amenity_imp.copy()
    table_df["adoption_%"]  = table_df["feature"].map(lambda x: f"{adoption_rates.get(x,0)*100:.1f}%")
    table_df["importance"]  = table_df["importance"].map(lambda x: f"{x:.5f}")
    table_df.index         += 1
    table_df.columns        = ["Feature", "RF Importance", "Adoption %"]
    st.dataframe(table_df, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 3 — RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────
elif page == "💡 Recommendation Engine":
    st.markdown("# Amenity Recommendation Engine")
    st.markdown("##### Select the amenities your property currently has. We'll identify high-impact gaps.")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    st.markdown("### What amenities does your property currently offer?")
    all_am_display = sorted([a.replace(" ", " ").title() for a in kept_amenities])

    selected_display = st.multiselect(
        "Select all that apply:",
        options=all_am_display,
        default=["Tv", "Internet", "Smoke Detector", "Essentials", "Dryer", "Washer"],
        help="Select every amenity your property currently provides."
    )
    selected_raw = {a.lower() for a in selected_display}

    top_n = st.slider("Number of recommendations:", 3, 15, 7)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── RECOMMENDATION LOGIC ──────────────────────────────────
    # Post-processing wrapper on top of the existing importance analysis.
    # Filters to amenities the property is MISSING, ranks by importance.
    missing = amenity_imp[~amenity_imp["feature"].isin(selected_raw)].copy()
    missing["adoption_%"] = missing["feature"].map(lambda x: adoption_rates.get(x, 0))
    missing["opportunity_score"] = (
        missing["importance"] * (1 - missing["adoption_%"])  # high impact + low adoption = best opportunity
    )
    missing["expected_lift_pts"] = (missing["importance"] / max_imp * 2.5).round(3)

    q75 = amenity_imp["importance"].quantile(0.75)
    q40 = amenity_imp["importance"].quantile(0.40)
    def priority(imp):
        if imp >= q75: return "High"
        if imp >= q40: return "Medium"
        return "Low"
    missing["priority"] = missing["importance"].apply(priority)

    recs = missing.sort_values("opportunity_score", ascending=False).head(top_n).reset_index(drop=True)

    if len(recs) == 0:
        st.success("✅ Your property already has all tracked amenities!")
    else:
        st.markdown(f"### Top {len(recs)} Recommended Amenities")
        st.caption("Ranked by opportunity score (importance × how rare the amenity is). "
                   "All expected lift values are indicative estimates, not causal predictions.")

        color_map = {"High": PALETTE["primary"], "Medium": PALETTE["accent"], "Low": PALETTE["muted"]}
        badge_map = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}

        for i, row in recs.iterrows():
            priority_color = color_map[row["priority"]]
            st.markdown(f"""
            <div class="rec-card" style="--priority-color:{priority_color}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-size:1.05rem;font-weight:600;color:{PALETTE['secondary']}">
                    {i+1}. {row['feature'].title()}
                  </span>
                  <span class="{badge_map[row['priority']]}" style="margin-left:10px;font-size:0.82rem">
                    ● {row['priority']} Priority
                  </span>
                </div>
                <div style="text-align:right">
                  <span style="font-size:0.8rem;color:{PALETTE['muted']}">
                    Currently in <strong>{row['adoption_%']*100:.0f}%</strong> of listings
                  </span>
                </div>
              </div>
              <div style="margin-top:8px;display:flex;gap:24px">
                <div>
                  <span style="font-size:0.75rem;color:{PALETTE['muted']};text-transform:uppercase;letter-spacing:0.06em">Importance Score</span><br>
                  <span style="font-weight:600;color:{PALETTE['secondary']}">{row['importance']:.5f}</span>
                </div>
                <div>
                  <span style="font-size:0.75rem;color:{PALETTE['muted']};text-transform:uppercase;letter-spacing:0.06em">Est. Rating Lift</span><br>
                  <span style="font-weight:600;color:{priority_color}">+{row['expected_lift_pts']} pts</span>
                </div>
                <div>
                  <span style="font-size:0.75rem;color:{PALETTE['muted']};text-transform:uppercase;letter-spacing:0.06em">Opportunity Score</span><br>
                  <span style="font-weight:600;color:{PALETTE['secondary']}">{row['opportunity_score']:.5f}</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        # Radar chart — current vs recommended
        st.markdown("### Coverage Radar")
        top8_amenities = amenity_imp.head(8)["feature"].tolist()
        has_vals  = [1 if a in selected_raw else 0 for a in top8_amenities]
        rec_set   = set(recs["feature"].tolist())
        after_vals = [1 if (a in selected_raw or a in rec_set) else 0 for a in top8_amenities]
        labels = [a.title() for a in top8_amenities]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=has_vals + [has_vals[0]], theta=labels + [labels[0]],
            fill="toself", name="Current", line_color=PALETTE["muted"], fillcolor=f"rgba(149,165,166,0.2)"))
        fig.add_trace(go.Scatterpolar(r=after_vals + [after_vals[0]], theta=labels + [labels[0]],
            fill="toself", name="After Recommendations", line_color=PALETTE["primary"], fillcolor=f"rgba(192,57,43,0.15)"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans"),
            height=380,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='warning-box'>
    ⚠️ <strong>Causal disclaimer:</strong> Recommendations are based on statistical association,
    not causal inference. Properties with certain amenities tend to have higher review scores,
    but adding those amenities does not guarantee a score improvement.
    The "expected lift" is an illustrative estimate only.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — SCORE PREDICTOR
# ─────────────────────────────────────────────────────────────
elif page == "🔮 Score Predictor":
    st.markdown("# Review Score Predictor")
    st.markdown("##### Configure a property and get a predicted review score from the trained model.")
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Property Details")
        accommodates  = st.slider("Accommodates (guests)", 1, 16, 4)
        bedrooms      = st.slider("Bedrooms", 0, 10, 2)
        beds          = st.slider("Beds", 1, 12, 2)
        bathrooms     = st.slider("Bathrooms", 0.5, 8.0, 1.0, step=0.5)
        room_type     = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
        property_type = st.selectbox("Property Type", ["House", "Apartment", "Condominium", "Townhouse", "Loft", "Other"])
        min_nights    = st.slider("Minimum Nights", 1, 30, 2)
        max_nights    = st.slider("Maximum Nights", 1, 365, 30)
        instant_book  = st.selectbox("Instant Bookable", ["Yes", "No"])

    with col2:
        st.markdown("#### Amenities Offered")
        am_display = sorted([a.title() for a in kept_amenities])
        defaults_on = ["Carbon Monoxide Detector", "Smoke Detector", "Essentials",
                       "Dryer", "Washer", "Internet", "Tv", "Shampoo"]
        selected_ams = st.multiselect("Select all amenities:", am_display, default=defaults_on)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    if st.button("🔮 Predict Review Score", type="primary"):
        # Build feature vector matching training schema exactly
        rt_map = {"Entire home/apt": 0, "Private room": 1, "Shared room": 2}
        pt_map = {"House": 0, "Apartment": 1, "Condominium": 2, "Townhouse": 3, "Loft": 4, "Other": 5}

        row = {
            "accommodates":   accommodates,
            "bedrooms":       bedrooms,
            "beds":           beds,
            "bathrooms":      bathrooms,
            "room_type":      rt_map.get(room_type, 0),
            "property_type":  pt_map.get(property_type, 0),
            "minimum_nights": min_nights,
            "maximum_nights": max_nights,
            "instant_bookable": 1 if instant_book == "Yes" else 0,
        }
        sel_raw = {a.lower() for a in selected_ams}
        for a in kept_amenities:
            row[a] = 1 if a.lower() in sel_raw else 0

        # Align to exact feature order
        feat_vector = np.array([[row.get(f, 0) for f in feature_names]])
        pred = float(np.clip(model.predict(feat_vector)[0], 20, 100))

        # Display
        c1, c2, c3 = st.columns(3)
        metric_card("Predicted Score", f"{pred:.1f} / 100", c1)
        metric_card("Dataset Mean", f"{y_dev_mean:.1f}", c2)
        metric_card("vs Mean", f"{pred - y_dev_mean:+.1f} pts", c3)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            delta={"reference": y_dev_mean, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [60, 100], "tickwidth": 1},
                "bar":  {"color": PALETTE["primary"]},
                "steps": [
                    {"range": [60, 80], "color": "#FADBD8"},
                    {"range": [80, 90], "color": "#FAE5D3"},
                    {"range": [90, 100],"color": "#D5F5E3"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["secondary"], "width": 3},
                    "thickness": 0.75,
                    "value": y_dev_mean,
                },
            },
            title={"text": "Predicted Review Score", "font": {"size": 16}},
            number={"suffix": " / 100", "font": {"size": 28}},
        ))
        fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans"))
        st.plotly_chart(fig, use_container_width=True)

        # Quick recommendations for missing high-impact amenities
        missing_high = amenity_imp[
            (amenity_imp["importance"] >= amenity_imp["importance"].quantile(0.75)) &
            (~amenity_imp["feature"].isin(sel_raw))
        ]["feature"].head(3).tolist()

        if missing_high:
            st.info(
                f"💡 **Quick wins:** Adding **{', '.join(a.title() for a in missing_high)}** "
                f"could positively associate with higher-rated listings based on the model."
            )

    st.markdown("""
    <div class='warning-box' style='margin-top:16px'>
    ⚠️ Predictions use the trained Random Forest model. The model explains ~4% of score variance (R²=0.04),
    reflecting the inherently difficult nature of predicting highly-compressed, left-skewed review scores.
    Use predictions directionally, not as precise estimates.
    </div>
    """, unsafe_allow_html=True)

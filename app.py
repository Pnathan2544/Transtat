import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangkok Rail Passenger Analytics",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
RAIL_LINES_TH = [
    "รถไฟฟ้า ARL", "รถไฟฟ้า BTS", "รถไฟฟ้าสายสีชมพู",
    "รถไฟฟ้าสายสีน้ำเงิน", "รถไฟฟ้าสายสีม่วง",
    "รถไฟฟ้าสายสีเหลือง", "รถไฟฟ้าสายสีแดง",
]
LINE_MAP = {
    "รถไฟฟ้า ARL":          "ARL",
    "รถไฟฟ้า BTS":          "BTS",
    "รถไฟฟ้าสายสีชมพู":     "MRT Pink",
    "รถไฟฟ้าสายสีน้ำเงิน":  "MRT Blue",
    "รถไฟฟ้าสายสีม่วง":     "MRT Purple",
    "รถไฟฟ้าสายสีเหลือง":   "MRT Yellow",
    "รถไฟฟ้าสายสีแดง":      "Red Line",
}
MODE_GROUP_MAP = {
    "ARL": "ARL", "BTS": "BTS",
    "MRT Blue": "MRT", "MRT Purple": "MRT",
    "MRT Yellow": "MRT", "MRT Pink": "MRT",
    "Red Line": "Red Line",
}
STATIONS = ["BTS", "MRT Blue", "MRT Pink", "MRT Purple", "MRT Yellow", "ARL", "Red Line"]
MODES    = ["BTS", "MRT", "ARL", "Red Line"]
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MODE_COLORS = {
    "BTS": "#1700AD", "MRT": "#06E061", "ARL": "#FF75D6", "Red Line": "#E00615",
}
STATION_COLORS = {
    "BTS": "#1700AD", "MRT Blue": "#065AE0", "MRT Pink": "#FF69BE",
    "MRT Purple": "#7F1894", "MRT Yellow": "#FFC300",
    "ARL": "#FF75D6", "Red Line": "#E00615",
}

THAI_EVENTS = {
    "2025-01-01": "วันปีใหม่",
    "2025-02-12": "วันมาฆบูชา",
    "2025-04-06": "วันจักรี",
    "2025-04-13": "วันสงกรานต์",
    "2025-05-01": "วันแรงงาน",
    "2025-05-05": "วันฉัตรมงคล",
    "2025-05-11": "วันวิสาขบูชา",
    "2025-06-03": "วันเฉลิมฯ สมเด็จพระราชินี",
    "2025-07-10": "วันอาสาฬหบูชา",
    "2025-07-11": "วันเข้าพรรษา",
    "2025-07-28": "วันเฉลิมฯ รัชกาลที่ 10",
    "2025-08-12": "วันแม่แห่งชาติ",
    "2025-10-13": "วันคล้ายวันสวรรคต ร.9",
    "2025-10-23": "วันปิยมหาราช",
    "2025-12-05": "วันพ่อแห่งชาติ",
    "2025-12-10": "วันรัฐธรรมนูญ",
    "2025-12-31": "วันส่งท้ายปีเก่า",
    "2026-01-01": "วันปีใหม่ 2026",
}

# ── Data Loading & Cleaning ───────────────────────────────────────────────────
def clean_volume(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d.]", "", regex=True)
        .str.strip()
    )

@st.cache_data(show_spinner="Loading & cleaning data...")
def load_data(path_2025: str, path_2026: str) -> pd.DataFrame:
    rename_map = {
        "รูปแบบการเดินทาง": "travel_mode",
        "วัตถุประสงค์": "purpose",
        "สาธารณะ/ส่วนบุคคล": "transport_type",
        "หน่วยงาน": "agency",
        "ยานพาหนะ/ท่า": "vehicle_station",
        "วันที่": "date",
        "หน่วย": "unit",
        "ปริมาณ": "volume",
    }
    df25 = pd.read_csv(path_2025, dtype=str).rename(columns=rename_map)
    df26 = pd.read_csv(path_2026, dtype=str).rename(columns=rename_map)

    for df, yr in [(df25, 2025), (df26, 2026)]:
        df.dropna(how="all", inplace=True)
        df["volume"] = pd.to_numeric(clean_volume(df["volume"]), errors="coerce")
        df["date"]   = pd.to_datetime(df["date"], errors="coerce")
        df["year"]   = yr

    df_all = (
        pd.concat([df25, df26], ignore_index=True)
        .sort_values("date")
        [["date", "year", "travel_mode", "purpose", "transport_type",
          "vehicle_station", "agency", "unit", "volume"]]
        .drop_duplicates()
    )
    return df_all

@st.cache_data(show_spinner="Engineering features...")
def build_analysis(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all[
        df_all["vehicle_station"].isin(RAIL_LINES_TH)
        & df_all["date"].notna()
        & df_all["volume"].notna()
        & (df_all["volume"] > 0)
    ].copy()

    df["line_en"]    = df["vehicle_station"].map(LINE_MAP)
    df["mode_group"] = df["line_en"].map(MODE_GROUP_MAP)

    # Fix month/day swap for invalid dates
    df["date"] = df["date"].apply(
        lambda d: d.replace(month=d.day, day=d.month) if pd.notna(d) and d.day <= 12 else d
    )
    df["year"]         = df["date"].dt.year
    df["weekday"]      = df["date"].dt.weekday
    df["weekday_name"] = df["date"].dt.day_name()
    df["is_weekend"]   = df["weekday"] >= 5
    df["month"]        = df["date"].dt.month

    return df.sort_values(["line_en", "date"]).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_daily(_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        _df.groupby(["date", "line_en"])["volume"]
        .sum().reset_index().sort_values(["line_en", "date"])
    )
    daily["year"]       = daily["date"].dt.year
    daily["dow"]        = daily["date"].dt.day_name()
    daily["is_weekend"] = daily["date"].dt.weekday >= 5
    daily["vol_minmax"] = daily.groupby("line_en")["volume"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    return daily

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚇 Bangkok Rail Analytics")
    st.caption("ข้อมูลผู้โดยสารระบบขนส่งทางรางในกรุงเทพฯ ปี 2025–2026")
    st.divider()

    st.subheader("📂 Data Files")
    path_2025 = st.text_input("passengers68.csv path", value="passengers68.csv")
    path_2026 = st.text_input("passengers69.csv path", value="passengers69.csv")

    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Q1 Modal Share", "📈 Q2 Line Comparison", "📅 Q3 Holiday Effects"],
        label_visibility="collapsed",
    )

# ── Load Data ─────────────────────────────────────────────────────────────────
try:
    df_all = load_data(path_2025, path_2026)
    df     = build_analysis(df_all)
    daily  = build_daily(df)
except FileNotFoundError as e:
    st.error(f"**File not found:** {e}\n\nPlace your CSV files in the same folder as `app.py`, or update the paths in the sidebar.")
    st.stop()

df_25    = df[df.year == 2025]
df_26    = df[df.year == 2026]
months_25 = df_25["date"].dt.to_period("M").nunique()
months_26 = df_26["date"].dt.to_period("M").nunique()

mode_vol_25 = df_25.groupby("mode_group")["volume"].sum().reindex(MODES)
mode_vol_26 = df_26.groupby("mode_group")["volume"].sum().reindex(MODES)
mode_pct_25 = mode_vol_25 / mode_vol_25.sum() * 100
mode_pct_26 = mode_vol_26 / mode_vol_26.sum() * 100

sta_vol_25 = df_25.groupby("line_en")["volume"].sum()
sta_vol_26 = df_26.groupby("line_en")["volume"].sum()
sta_pct_25 = sta_vol_25 / sta_vol_25.sum() * 100
sta_pct_26 = sta_vol_26 / sta_vol_26.sum() * 100

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Bangkok Public Rail — Passenger Analytics")
    st.caption(f"Data: THackle / BDI | Coverage: {df['date'].min().date()} → {df['date'].max().date()}")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    total_25 = df_25["volume"].sum()
    total_26 = df_26["volume"].sum()
    c1.metric("Total Riders 2025", f"{total_25/1e6:.1f}M", f"{months_25} months")
    c2.metric("Total Riders 2026", f"{total_26/1e6:.1f}M", f"{months_26} months")
    c3.metric("Lines Tracked", len(STATIONS))
    c4.metric("Data Points", f"{len(df):,}")

    st.divider()
    st.subheader("Research Questions")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Q1 — Modal Share**\nคนไทยเดินทางด้วยอะไรมากที่สุด?\nWhat do Bangkokians use most?")
    with col2:
        st.info("**Q2 — Line Comparison**\nแต่ละสายมีพฤติกรรมต่างกันอย่างไร?\nHow do lines differ in ridership patterns?")
    with col3:
        st.info("**Q3 — Holiday Effects**\nวันหยุดเห็นได้ในข้อมูลไหม?\nAre holidays & festivals visible in the data?")

    st.divider()
    st.subheader("Coverage by Line")
    cov = (
        df.groupby("line_en").agg(
            Start=("date", "min"), End=("date", "max"),
            Days=("date", "nunique"),
            Total_Riders=("volume", "sum"),
            Avg_Daily=("volume", "mean"),
        )
        .reindex(STATIONS)
        .round(0).astype({"Days": int, "Total_Riders": int, "Avg_Daily": int})
    )
    cov["Total_Riders"] = cov["Total_Riders"].map("{:,}".format)
    cov["Avg_Daily"]    = cov["Avg_Daily"].map("{:,}".format)
    st.dataframe(cov, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Q1 — MODAL SHARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Q1 Modal Share":
    st.title("Q1 — Modal Share")
    st.markdown("**คนไทยเดินทางด้วยอะไรมากที่สุด?** Which rail system carries the most passengers?")
    st.divider()

    # Chart 1 — Donut
    fig1 = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=[f"2025 ({months_25} months)", f"2026 ({months_26} months)"],
    )
    for col, pct in enumerate([mode_pct_25, mode_pct_26], 1):
        fig1.add_trace(go.Pie(
            labels=MODES, values=pct.round(2).values,
            hole=0.55,
            marker=dict(colors=[MODE_COLORS[m] for m in MODES]),
            textinfo="label+percent",
            insidetextorientation="radial",
            showlegend=(col == 1),
        ), row=1, col=col)
    fig1.update_layout(
        title_text="<b>Modal Share — Bangkok Urban Rail Systems</b>",
        height=420, margin=dict(t=80, b=60, r=150),
        legend=dict(orientation="v", x=1.05, y=0.5),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.info("💡 BTS holds ~50% modal share; MRT (all lines combined) ~42% and growing; ARL + Red Line < 8%.")

    st.divider()

    # Chart 2 — Grouped Bar
    bar_df = pd.DataFrame({
        "Station": STATIONS * 2,
        "Year":    ["2025"] * len(STATIONS) + ["2026"] * len(STATIONS),
        "Share":   list(sta_pct_25.reindex(STATIONS).values) + list(sta_pct_26.reindex(STATIONS).values),
    })
    fig2 = px.bar(
        bar_df, x="Station", y="Share", color="Year", barmode="group",
        color_discrete_map={"2025": "#4A90D9", "2026": "#E87040"},
        text=bar_df["Share"].apply(lambda v: f"{v:.2f}%"),
        labels={"Share": "Share of Total Riders (%)"},
        title="<b>Passenger Share per Line — 2025 vs 2026</b>",
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        yaxis=dict(range=[0, bar_df.Share.max() * 1.2]),
        height=460, margin=dict(t=70, b=60),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Chart 3 — Diverging Bar
    div_df = pd.DataFrame({
        "Station": STATIONS,
        "Avg_25":  [sta_pct_25.get(s, 0) for s in STATIONS],
        "Avg_26":  [sta_pct_26.get(s, 0) for s in STATIONS],
    }).assign(Change=lambda d: d.Avg_26 - d.Avg_25).sort_values("Change", ascending=True)
    div_df["Color"] = div_df["Change"].apply(lambda x: "#2ECC71" if x >= 0 else "#E74C3C")

    fig3 = go.Figure(go.Bar(
        x=div_df["Change"], y=div_df["Station"], orientation="h",
        marker_color=div_df["Color"],
        text=div_df["Change"].apply(lambda v: f"{v:+.2f}"),
        textposition="outside",
    ))
    fig3.add_vline(x=0, line_width=1.5, line_color="gray")
    fig3.update_layout(
        title=f"<b>Change in Passenger Share</b><br>"
              f"<sup>2025 ({months_25} mo) → 2026 ({months_26} mo) | Green = gained | Red = lost</sup>",
        xaxis_title="Δ Share (%)", height=420,
        margin=dict(t=90, b=60, l=130, r=100),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.info("💡 MRT Blue and ARL gained share in 2026; BTS and MRT Purple slightly declined.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Q2 — LINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Q2 Line Comparison":
    st.title("Q2 — Line Comparison")
    st.markdown("**แต่ละสายมีพฤติกรรมต่างกันอย่างไร?** How do ridership patterns differ across lines?")
    st.divider()

    year_filter = st.selectbox("Year", [2025, 2026], index=0)
    st.divider()

    # Box Plots (Min-Max Scaled)
    st.subheader("Daily Ridership Distribution (Min-Max Scaled)")
    fig_box = make_subplots(
        rows=2, cols=4, subplot_titles=STATIONS,
        vertical_spacing=0.18, horizontal_spacing=0.08,
    )
    for i, sta in enumerate(STATIONS):
        row, col = i // 4 + 1, i % 4 + 1
        sub = daily[(daily.line_en == sta) & (daily.year == year_filter)]
        fig_box.add_trace(go.Box(
            y=sub["vol_minmax"],
            name=sta,
            marker_color=STATION_COLORS[sta],
            boxmean="sd",
            width=0.6,
            showlegend=False,
            customdata=sub["volume"].values,
            hovertemplate=f"<b>{sta}</b><br>Scaled: %{{y:.3f}}<br>Actual: %{{customdata:,.0f}}<extra></extra>",
        ), row=row, col=col)
    fig_box.update_layout(
        title_text=f"<b>Daily Ridership Distribution — {year_filter} (Min-Max Scaled)</b>",
        height=580, margin=dict(t=90, b=60),
    )
    for r in [1, 2]:
        fig_box.update_yaxes(title_text="Scaled Volume (0–1)", row=r, col=1)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # CV Chart
    st.subheader("Coefficient of Variation (Volatility)")
    cv_rows = []
    for sta in STATIONS:
        for yr in [2025, 2026]:
            vals = daily[(daily.line_en == sta) & (daily.year == yr)]["volume"]
            if len(vals) < 2:
                continue
            cv_rows.append({"Station": sta, "Year": str(yr), "CV": vals.std() / vals.mean() * 100})
    cv_df = pd.DataFrame(cv_rows)

    fig_cv = px.bar(
        cv_df, x="Station", y="CV", color="Year", barmode="group",
        color_discrete_map={"2025": "#4A90D9", "2026": "#E87040"},
        text=cv_df["CV"].apply(lambda v: f"{v:.1f}%"),
        title="<b>Coefficient of Variation (CV) per Line</b><br>"
              "<sup>Higher CV = more volatile | Lower CV = stable ridership base</sup>",
        labels={"CV": "CV (%)"},
        category_orders={"Station": STATIONS},
    )
    fig_cv.update_traces(textposition="outside")
    fig_cv.update_layout(
        yaxis=dict(range=[0, cv_df.CV.max() * 1.25]),
        height=440, margin=dict(t=80, b=60),
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    st.info("💡 ARL is most stable (CV ~17%): consistent airport demand. MRT Purple most volatile (CV ~30%): heavy commuter dependency.")

    st.divider()

    # Monthly Heatmap
    st.subheader("Monthly Trend Heatmap (% of Line's Own Average)")
    heat_rows = []
    for sta in STATIONS:
        sub = (
            daily[daily.line_en == sta]
            .set_index("date")["volume"]
            .sort_index()
            .rolling("30D", min_periods=7).mean()
            .resample("ME").mean()
        )
        for dt, val in sub.items():
            heat_rows.append({"Station": sta, "Month": dt, "AvgVol": val})

    heat_df   = pd.DataFrame(heat_rows)
    pivot_h   = heat_df.pivot(index="Station", columns="Month", values="AvgVol").reindex(index=STATIONS)
    pivot_n   = pivot_h.div(pivot_h.mean(axis=1), axis=0) * 100

    fig_heat = go.Figure(go.Heatmap(
        z=pivot_n.values,
        x=[m.strftime("%b %Y") for m in pivot_h.columns],
        y=STATIONS,
        colorscale="RdYlGn",
        zmid=100,
        text=np.round(pivot_h.values / 1000, 1),
        texttemplate="%{text}k",
        hovertemplate="<b>%{y}</b> — %{x}<br>vs own avg: %{z:.1f}%<br>Actual: %{text}k<extra></extra>",
        colorbar=dict(title="% of line's avg", tickvals=[80, 90, 100, 110, 120],
                      ticktext=["80%", "90%", "avg", "110%", "120%"]),
    ))
    fig_heat.update_layout(
        title="<b>30-Day Rolling Avg Passenger Trend — Jan 2025 to Mar 2026</b><br>"
              "<sup>Green = above line average | Red = below average</sup>",
        xaxis=dict(tickangle=-45),
        height=420, margin=dict(t=80, b=100, l=130, r=80),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.info("💡 June–August 2025: below-average ridership across almost all lines (rainy season + post-summer break).")

    st.divider()

    # Day-of-Week Heatmap
    st.subheader("Day-of-Week Ridership Pattern")
    pivot_dow = (
        daily.groupby(["line_en", "dow"])["volume"]
        .mean().reset_index()
        .pivot(index="line_en", columns="dow", values="volume")
        .reindex(index=STATIONS, columns=DOW_ORDER)
    )
    pivot_dow_n = pivot_dow.div(pivot_dow.mean(axis=1), axis=0) * 100

    fig_dow = go.Figure(go.Heatmap(
        z=pivot_dow_n.values,
        x=[d[:3] for d in DOW_ORDER],
        y=STATIONS,
        colorscale="RdYlGn",
        zmid=100,
        text=np.round(pivot_dow.values / 1000, 1),
        texttemplate="%{text}k",
        hovertemplate="<b>%{y}</b> — %{x}<br>vs weekly avg: %{z:.1f}%<br>Actual avg: %{text}k<extra></extra>",
        colorbar=dict(title="% of weekly avg"),
    ))
    fig_dow.update_layout(
        title="<b>Average Ridership by Day of Week</b><br>"
              "<sup>Green = above weekly avg | Red = below | Numbers in thousands</sup>",
        height=400, margin=dict(t=80, b=60, l=130),
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    st.info("💡 Mon–Fri ridership significantly higher than weekends. MRT Purple drops most sharply on weekends (suburban commuter line).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Q3 — HOLIDAY EFFECTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Q3 Holiday Effects":
    st.title("Q3 — Holiday Effects")
    st.markdown("**วันหยุดและเทศกาลเห็นได้ในข้อมูลไหม?** Are holidays & festivals visible in the data?")
    st.divider()

    # Total daily across all lines
    total_daily = (
        df.groupby("date")["volume"].sum()
        .reset_index().sort_values("date")
    )
    total_daily["rolling_mean"] = total_daily["volume"].rolling(7, center=True, min_periods=3).mean()
    total_daily["rolling_std"]  = total_daily["volume"].rolling(7, center=True, min_periods=3).std()
    total_daily["z_score"] = (
        (total_daily["volume"] - total_daily["rolling_mean"])
        / total_daily["rolling_std"].replace(0, np.nan)
    )

    z_thresh = st.slider("Anomaly Z-score threshold", 1.0, 3.0, 1.5, 0.1)
    anomaly_low  = total_daily[total_daily["z_score"] < -z_thresh]
    anomaly_high = total_daily[total_daily["z_score"] >  z_thresh]

    c1, c2 = st.columns(2)
    c1.metric("Low anomalies (holidays/dips)", len(anomaly_low))
    c2.metric("High anomalies (spikes)", len(anomaly_high))
    st.divider()

    # Time Series
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=total_daily["date"], y=total_daily["volume"],
        mode="lines", name="Daily Total",
        line=dict(color="#3498DB", width=1.2), opacity=0.7,
    ))
    fig_ts.add_trace(go.Scatter(
        x=total_daily["date"], y=total_daily["rolling_mean"],
        mode="lines", name="7-day Rolling Avg",
        line=dict(color="#E67E22", width=2, dash="dot"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=anomaly_low["date"], y=anomaly_low["volume"],
        mode="markers", name=f"Low anomaly (z < -{z_thresh})",
        marker=dict(color="#E74C3C", size=8, symbol="circle"),
    ))
    fig_ts.add_trace(go.Scatter(
        x=anomaly_high["date"], y=anomaly_high["volume"],
        mode="markers", name=f"High anomaly (z > +{z_thresh})",
        marker=dict(color="#27AE60", size=8, symbol="diamond"),
    ))

    for date_str, label in THAI_EVENTS.items():
        try:
            dt = pd.to_datetime(date_str)
            if total_daily["date"].min() <= dt <= total_daily["date"].max():
                fig_ts.add_vline(x=dt, line_width=1, line_color="gray", line_dash="dot", opacity=0.5)
                fig_ts.add_annotation(
                    x=dt, y=total_daily["volume"].max() * 0.95,
                    text=label, showarrow=False,
                    textangle=-90, font=dict(size=8, color="gray"), xshift=5,
                )
        except Exception:
            pass

    fig_ts.update_layout(
        title="<b>Daily Ridership — All Rail Lines Combined (Jan 2025 – Mar 2026)</b><br>"
              "<sup>Red = low anomaly | Green = high anomaly | Vertical lines = Thai holidays</sup>",
        xaxis_title="Date", yaxis_title="Total Riders",
        height=520, margin=dict(t=100, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.divider()

    # Holiday Volume Table
    st.subheader("Ridership Near Each Holiday")
    report_rows = []
    for date_str, event_name in THAI_EVENTS.items():
        event_dt = pd.to_datetime(date_str)
        tmp = total_daily.copy()
        tmp["diff"] = (tmp["date"] - event_dt).abs()
        closest = tmp.nsmallest(2, "diff").nsmallest(1, "volume").iloc[0]
        report_rows.append({
            "Event Date": event_dt.strftime("%Y-%m-%d"),
            "Event": event_name,
            "Closest Data Date": closest["date"].strftime("%Y-%m-%d"),
            "Riders": int(closest["volume"]),
            "Days Apart": closest["diff"].days,
        })

    holiday_df = pd.DataFrame(report_rows).sort_values("Event Date").reset_index(drop=True)
    holiday_df["Riders"] = holiday_df["Riders"].map("{:,}".format)
    st.dataframe(holiday_df, use_container_width=True)

    st.divider()

    # Calendar Heatmap — 2025
    st.subheader("Calendar Heatmap — 2025")
    cal_2025 = total_daily[total_daily["date"].dt.year == 2025].copy()
    cal_2025["week"]    = cal_2025["date"].dt.isocalendar().week.astype(int)
    cal_2025["dow_num"] = cal_2025["date"].dt.weekday

    pivot_cal = cal_2025.pivot_table(index="dow_num", columns="week", values="volume", aggfunc="mean")

    fig_cal = go.Figure(go.Heatmap(
        z=pivot_cal.values / 1000,
        x=[f"W{w}" for w in pivot_cal.columns],
        y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Week %{x} — %{y}<br>Avg riders: %{z:.0f}k<extra></extra>",
        colorbar=dict(title="Riders (k)", ticksuffix="k"),
    ))
    fig_cal.update_layout(
        title="<b>Calendar Heatmap — 2025</b><br>"
              "<sup>Darker = more riders | Blank = no data for that cell</sup>",
        xaxis_title="Week of Year", yaxis_title="Day",
        height=320, margin=dict(t=80, b=60),
    )
    st.plotly_chart(fig_cal, use_container_width=True)
    st.info("💡 Songkran week (mid-April) and New Year period show clear ridership drops across all lines.")

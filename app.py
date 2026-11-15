"""
NILM Smart Energy - Streamlit website.

Run:  streamlit run app.py

Uses only saved artefacts: processed data (data/processed), trained models
(models/*.pth) and evaluation results (results/). Nothing is retrained here.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_utils as U
from src.dataset import NILMData
from src.utils import (MODEL_NAMES, MODEL_LABELS, APPLIANCE_NAMES, MIN_TARGET, HIGH_TARGET,
                       FIG_DIR, REGION, BENCHMARK_APPLIANCES)

st.set_page_config(page_title="NILM Smart Energy", page_icon="⚡", layout="wide")

TEAL, BLUE, GREEN, GREY, AMBER = "#14b8a6", "#3b82f6", "#22c55e", "#64748b", "#f59e0b"
ICONS = {"air1": "❄️", "drye1": "🌀", "car1": "🔌", "refrigerator1": "🧊"}
NOTE = ("Historical Dataset Demonstration — Predictions are generated using trained ANN "
        "models on historical household power readings.")

st.markdown("""
<style>
.stApp {background: #0b1220; color: #e2e8f0;}
section[data-testid="stSidebar"] {background: #0f172a;}
h1, h2, h3 {color: #f1f5f9; font-weight: 650;}
.card {background: #111c2e; border: 1px solid #1e293b; border-radius: 14px;
       padding: 16px 18px; margin-bottom: 10px;}
.card .label {color: #94a3b8; font-size: 0.80rem; text-transform: uppercase;
              letter-spacing: .04em;}
.card .value {color: #f8fafc; font-size: 1.6rem; font-weight: 700; margin-top: 4px;}
.app {display:flex; justify-content:space-between; align-items:center;}
.badge {padding: 3px 10px; border-radius: 999px; font-size: .75rem; font-weight: 700;}
.on  {background: rgba(34,197,94,.15); color: #4ade80; border: 1px solid #16a34a;}
.off {background: rgba(100,116,139,.15); color: #94a3b8; border: 1px solid #475569;}
.na  {background: rgba(245,158,11,.12); color: #fbbf24; border: 1px solid #b45309;}
.note {background:#0f2a2a; border-left: 4px solid #14b8a6; padding: 8px 12px;
       border-radius: 6px; color:#99f6e4; font-size: .85rem; margin-bottom: 14px;}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- cached loaders
@st.cache_resource
def get_data():
    return NILMData()


@st.cache_resource
def get_model(name):
    return U.load_torch_model(name, get_data())


@st.cache_data
def get_results():
    return U.load_results()


def plot_layout(fig, title="", height=330, ytitle="Power (W)"):
    fig.update_layout(title=title, height=height, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title=ytitle,
                      legend=dict(orientation="h", y=1.12))
    return fig


def metric_card(col, label, value):
    col.markdown(f'<div class="card"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def appliance_card(col, a, on, watts, actual_w=None, metered=True):
    if not metered:
        badge, txt = "na", "NOT METERED"
    else:
        badge, txt = ("on", "ON") if on else ("off", "OFF")
    extra = "" if actual_w is None or np.isnan(actual_w) else \
        f'<div class="label" style="margin-top:6px">actual reading: {actual_w:,.0f} W</div>'
    pw = "-" if not metered else f"{watts:,.0f} W"
    col.markdown(f'''<div class="card"><div class="app">
      <div class="label">{ICONS.get(a, "")} {APPLIANCE_NAMES[a]}</div>
      <span class="badge {badge}">{txt}</span></div>
      <div class="value">{pw}</div>
      <div class="label">predicted power</div>{extra}</div>''', unsafe_allow_html=True)


def run_prediction(name, hi, ends):
    """Returns (prob, watts, on, message). prob/watts None when the model cannot run."""
    data = get_data()
    model, msg = get_model(name)
    if model is None:
        return None, None, None, msg
    prob, watts = U.predict_ends(model, data, hi, np.asarray(ends))
    cut = U.prob_cutoffs(get_results(), name, data.appliances)
    on = prob >= np.array([cut[a] for a in data.appliances])
    return prob, watts, on, msg


# ---------------------------------------------------------------- sidebar
data = get_data()
res = get_results()
status = st.cache_data(ttl=60)(U.model_status)()

st.sidebar.markdown("## ⚡ NILM Smart Energy")
st.sidebar.caption("AI-Powered Non-Intrusive Load Monitoring System")
page = st.sidebar.radio("Navigation", ["Home Dashboard", "Appliance Monitoring",
                                       "Model Comparison", "Prediction Demo",
                                       "Model Evaluation", "Historical Analysis"])
home = st.sidebar.selectbox(
    "Home", data.homes,
    format_func=lambda h: f"Home {h}" + (" (unseen)" if h in data.prep["unseen_homes"] else ""))
model_name = st.sidebar.selectbox(
    "ANN model", MODEL_NAMES,
    format_func=lambda n: MODEL_LABELS[n] + (" ✓" if status.set_index("name").loc[n, "exists"]
                                             else " (not trained)"))
hi = U.home_index(data, home)
trained = int(status.exists.sum())
st.sidebar.caption(f"{trained}/6 models trained · region: Austin, TX")

st.title("NILM Smart Energy")
st.caption("AI-Powered Non-Intrusive Load Monitoring System")
st.markdown(f'<div class="note">{NOTE}</div>', unsafe_allow_html=True)
metered = data.prep["availability"][str(home)]


# ================================================================ PAGE 1
if page == "Home Dashboard":
    end = U.default_sample(data, hi)
    t_end = U.local_times(data, hi, end, end + 1)[0]
    split = U.SPLIT_NAMES[int(data.h[hi]["split"][end])]
    st.subheader(f"Home {home} · {t_end:%d %b %Y, %H:%M} ({split} data)")
    prob, watts, on, msg = run_prediction(model_name, hi, [end])
    agg_w = float(data.h[hi]["aggregate"][end] * 1000)
    c = st.columns(4)
    metric_card(c[0], "Total home power", f"{agg_w:,.0f} W")
    metric_card(c[1], "Monitored appliances", sum(metered.values()))
    metric_card(c[2], "Appliances ON (predicted)",
                "-" if on is None else int(sum(on[0][i] for i, a in
                                               enumerate(data.appliances) if metered[a])))
    metric_card(c[3], "Selected ANN model", MODEL_LABELS[model_name])
    if msg:
        st.warning(msg)

    cols = st.columns(len(data.appliances))
    for i, a in enumerate(data.appliances):
        actual = data.h[hi]["power"][end, i] * 1000
        if prob is None and metered[a]:
            cols[i].markdown(f'<div class="card"><div class="label">{APPLIANCE_NAMES[a]}</div>'
                             '<div class="value">-</div><div class="label">model not available'
                             '</div></div>', unsafe_allow_html=True)
        elif prob is None:
            appliance_card(cols[i], a, False, np.nan, None, metered=False)
        else:
            appliance_card(cols[i], a, bool(on[0, i]), float(watts[0, i]), actual, metered[a])

    df = U.range_frame(data, hi, max(0, end - 1440 + 1), end + 1)
    fig = go.Figure(go.Scatter(x=df.time, y=df.aggregate_W, line=dict(color=TEAL),
                               fill="tozeroy", name="whole-house power"))
    st.plotly_chart(plot_layout(fig, "Total household power, last 24 hours (historical)"),
                    use_container_width=True)
    if prob is not None:
        bar = go.Figure()
        names = [APPLIANCE_NAMES[a] for a in data.appliances]
        bar.add_bar(x=names, y=watts[0], name="predicted", marker_color=TEAL)
        bar.add_bar(x=names, y=[data.h[hi]["power"][end, i] * 1000
                                for i in range(len(names))], name="actual", marker_color=GREY)
        st.plotly_chart(plot_layout(bar, "Appliance power overview at this minute"),
                        use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Home {home}** · {'unseen home (never used in training)' if home in data.prep['unseen_homes'] else 'development home'}  \n"
                f"Metered appliances: {', '.join(APPLIANCE_NAMES[a] for a in data.appliances if metered[a])}")
    s = status.set_index("name").loc[model_name]
    c2.markdown(f"**{MODEL_LABELS[model_name]}** · checkpoint "
                f"{'found, best epoch ' + str(s.best_epoch) if s.exists else 'missing'}  \n"
                f"Input: last {data.window} minutes of whole-house power only.")


# ================================================================ PAGE 2
elif page == "Appliance Monitoring":
    a = st.selectbox("Appliance", data.appliances, format_func=U.appliance_label)
    i = data.appliances.index(a)
    if not metered[a]:
        st.info(f"{APPLIANCE_NAMES[a]} is not metered in home {home}. Predictions are shown "
                "but there is no ground truth to compare with.")
    default = U.local_times(data, hi, U.default_sample(data, hi),
                            U.default_sample(data, hi) + 1)[0].date()
    day = st.date_input("Day", default)
    a0 = U.minute_at(data, hi, pd.Timestamp(day))
    b0 = min(a0 + 1440, len(data.h[hi]["ts"]))
    df = U.range_frame(data, hi, a0, b0)
    ends = U.valid_ends_in(data, hi, a0, b0)
    prob, watts, on, msg = run_prediction(model_name, hi, ends) if len(ends) else (None,) * 3 + ("No valid input windows on this day.",)
    if msg:
        st.warning(msg)
    if prob is not None:
        tp = U.local_times(data, hi)[ends]
        last = -1
        c = st.columns(3)
        metric_card(c[0], "Latest predicted status", "ON" if on[last, i] else "OFF")
        metric_card(c[1], "Latest predicted power", f"{watts[last, i]:,.0f} W")
        act = df[f"{a}_W"].iloc[-1]
        metric_card(c[2], "Latest actual power", "-" if np.isnan(act) else f"{act:,.0f} W")
    fig = go.Figure()
    fig.add_scatter(x=df.time, y=df[f"{a}_W"], name="actual (historical reading)",
                    line=dict(color="#e2e8f0", width=1.2))
    if prob is not None:
        fig.add_scatter(x=tp, y=watts[:, i], name=f"predicted ({MODEL_LABELS[model_name]})",
                        line=dict(color=TEAL, width=1.2))
    st.plotly_chart(plot_layout(fig, f"{APPLIANCE_NAMES[a]}: actual vs predicted power"),
                    use_container_width=True)
    fig = go.Figure()
    fig.add_scatter(x=df.time, y=df[f"{a}_on"], name="actual ON/OFF", line_shape="hv",
                    line=dict(color="#e2e8f0"))
    if prob is not None:
        fig.add_scatter(x=tp, y=on[:, i].astype(int) * 0.9, name="predicted ON/OFF (×0.9)",
                        line_shape="hv", line=dict(color=TEAL))
    st.plotly_chart(plot_layout(fig, "ON/OFF status history", 260, "state (1 = ON)"),
                    use_container_width=True)
    wk_a = max(0, a0 - 6 * 1440)
    hist = U.range_frame(data, hi, wk_a, b0)
    fig = go.Figure(go.Scatter(x=hist.time, y=hist[f"{a}_W"], line=dict(color=BLUE, width=1),
                               name="historical reading"))
    st.plotly_chart(plot_layout(fig, "Historical appliance power, last 7 days (ground truth)"),
                    use_container_width=True)


# ================================================================ PAGE 3
elif page == "Model Comparison":
    comp = res["comparison"]
    st.dataframe(status[["label", "exists", "finished", "best_epoch"]].rename(
        columns={"label": "model", "exists": "checkpoint saved"}), hide_index=True)
    if comp is None:
        st.info("No evaluation results yet. Run `python -m src.train` and then "
                "`python -m src.evaluate`.")
    else:
        missing = [MODEL_LABELS[n] for n in MODEL_NAMES if MODEL_LABELS[n] not in set(comp.model)]
        if missing:
            st.warning("Not evaluated yet: " + ", ".join(missing))
        show = comp.copy()
        show["minimum target"] = show.meets_min.map({True: "✅ met", False: "❌ not met"})
        show["higher target"] = show.meets_high.map({True: "✅ met", False: "❌ not met"})
        st.markdown("#### Test-set results, all models")
        st.dataframe(show.drop(columns=["meets_min", "meets_high"]).round(3),
                     hide_index=True, use_container_width=True)
        metric = st.selectbox("Metric", ["accuracy", "balanced_accuracy", "macro_f1",
                                         "mae_W", "rmse_W"])
        fig = go.Figure()
        for m in comp.model.unique():
            d = comp[comp.model == m]
            fig.add_bar(x=[APPLIANCE_NAMES[a] for a in d.appliance], y=d[metric], name=m)
        if metric in MIN_TARGET:
            fig.add_hline(MIN_TARGET[metric], line_dash="dash", line_color=AMBER,
                          annotation_text="minimum target")
            fig.add_hline(HIGH_TARGET[metric], line_dash="dot", line_color=GREEN,
                          annotation_text="higher target")
        st.plotly_chart(plot_layout(fig, f"{metric} by appliance and model", 420, metric),
                        use_container_width=True)
        if res["best"] is not None:
            st.markdown("#### Best model per appliance and metric")
            st.dataframe(res["best"], hide_index=True)
        if res["bench"] is not None:
            st.markdown(f"#### CCP benchmark ({', '.join(BENCHMARK_APPLIANCES)})")
            st.dataframe(res["bench"], hide_index=True)
        if res["unseen"] is not None:
            st.markdown("#### Unseen-home results (homes never used in training)")
            st.dataframe(res["unseen"], hide_index=True, use_container_width=True)


# ================================================================ PAGE 4
elif page == "Prediction Demo":
    st.subheader("Run a trained model on one historical moment")
    default_end = U.default_sample(data, hi)
    t_def = U.local_times(data, hi, default_end, default_end + 1)[0]
    c1, c2 = st.columns(2)
    day = c1.date_input("Date", t_def.date())
    tm = c2.time_input("Time", t_def.time().replace(second=0))
    end = U.minute_at(data, hi, pd.Timestamp.combine(day, tm))
    split = U.SPLIT_NAMES[int(data.h[hi]["split"][end])]
    st.caption(f"This minute belongs to the **{split}** part of home {home}. "
               "Test and unseen-home minutes were never seen during training.")
    if st.button(f"Run {MODEL_LABELS[model_name]}", type="primary"):
        if not U.window_ok(data, hi, end):
            st.error("The 180-minute window before this time has missing readings. "
                     "Choose another time.")
        else:
            prob, watts, on, msg = run_prediction(model_name, hi, [end])
            if prob is None:
                st.error(msg)
            else:
                if msg:
                    st.warning(msg)
                cols = st.columns(len(data.appliances))
                for i, a in enumerate(data.appliances):
                    appliance_card(cols[i], a, bool(on[0, i]), float(watts[0, i]),
                                   data.h[hi]["power"][end, i] * 1000, metered[a])
                st.dataframe(pd.DataFrame({
                    "appliance": [APPLIANCE_NAMES[a] for a in data.appliances],
                    "ON probability": np.round(prob[0], 3),
                    "predicted power (W)": np.round(watts[0]),
                    "actual power (W)": np.round(data.h[hi]["power"][end] * 1000),
                }), hide_index=True)
            w = U.range_frame(data, hi, end - data.window + 1, end + 1)
            fig = go.Figure(go.Scatter(x=w.time, y=w.aggregate_W, fill="tozeroy",
                                       line=dict(color=TEAL), name="model input"))
            st.plotly_chart(plot_layout(fig, f"Model input: {data.window} minutes of "
                                             "whole-house power (the only input)"),
                            use_container_width=True)


# ================================================================ PAGE 5
elif page == "Model Evaluation":
    logs = res["logs"]
    if not logs:
        st.info("No training logs yet. Run `python -m src.train`.")
    else:
        cols = st.columns(3)
        for k, (n, h) in enumerate(logs.items()):
            fig = go.Figure()
            fig.add_scatter(x=h.epoch, y=h.train_loss, name="training", line=dict(color=BLUE))
            fig.add_scatter(x=h.epoch, y=h.val_loss, name="validation", line=dict(color=TEAL))
            cols[k % 3].plotly_chart(plot_layout(fig, MODEL_LABELS[n], 260, "loss"),
                                     use_container_width=True)
        st.caption("Validation loss is used for early stopping; the saved model is the "
                   "epoch with the lowest validation loss.")
    r = res["results"]
    if r is not None:
        sel = r[(r.split == "test") & (r.home == "all") & (r.model == MODEL_LABELS[model_name])]
        if sel.empty:
            st.info(f"{MODEL_LABELS[model_name]} has no test results yet.")
        else:
            st.markdown(f"#### {MODEL_LABELS[model_name]} on the test set")
            st.dataframe(sel[["appliance", "n_on", "n_off", "accuracy", "balanced_accuracy",
                              "macro_f1", "precision", "recall", "mae_W", "rmse_W",
                              "meets_min", "meets_high"]].round(3), hide_index=True,
                         use_container_width=True)
            cols = st.columns(len(sel))
            for col, (_, row) in zip(cols, sel.iterrows()):
                cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
                fig = go.Figure(go.Heatmap(z=cm, x=["pred OFF", "pred ON"],
                                           y=["actual OFF", "actual ON"], colorscale="Teal",
                                           text=cm, texttemplate="%{text:,}", showscale=False))
                col.plotly_chart(plot_layout(fig, APPLIANCE_NAMES[row.appliance], 260, ""),
                                 use_container_width=True)
    st.markdown("""
**How to read these numbers.** *Accuracy* is the share of minutes classified correctly.
For rare appliances such as the dryer it can look high even if every ON minute is missed,
so *balanced accuracy* (average of ON recall and OFF recall) and *macro-F1* (average F1 of
both classes) are the fairer measures. *MAE* and *RMSE* show the power error in watts;
RMSE punishes large misses more.""")
    for a in data.appliances:
        f = FIG_DIR / f"actual_vs_pred_power_{a}.png"
        if f.exists():
            st.image(str(f), caption=f"{APPLIANCE_NAMES[a]}: actual vs predicted power, all models")


# ================================================================ PAGE 6
elif page == "Historical Analysis":
    st.subheader("Historical ground-truth readings (not predictions)")
    t0 = U.local_times(data, hi, 0, 1)[0]
    c1, c2, c3 = st.columns(3)
    day = c1.date_input("Date", (t0 + pd.Timedelta(days=180)).date())
    hours = c2.slider("Time range (hours)", 0, 24, (0, 24))
    apps = c3.multiselect("Appliances", data.appliances, default=data.appliances,
                          format_func=U.appliance_label)
    a0 = U.minute_at(data, hi, pd.Timestamp(day) + pd.Timedelta(hours=hours[0]))
    b0 = U.minute_at(data, hi, pd.Timestamp(day) + pd.Timedelta(hours=hours[1])) + 1
    df = U.range_frame(data, hi, a0, b0)
    if df.empty:
        st.info("No readings in this range.")
    else:
        fig = go.Figure(go.Scatter(x=df.time, y=df.aggregate_W, line=dict(color=TEAL),
                                   fill="tozeroy", name="whole-house power"))
        st.plotly_chart(plot_layout(fig, "Total household power (historical)"),
                        use_container_width=True)
        fig = go.Figure()
        for a in apps:
            fig.add_scatter(x=df.time, y=df[f"{a}_W"], name=APPLIANCE_NAMES[a], line=dict(width=1))
        st.plotly_chart(plot_layout(fig, "Appliance power (historical ground truth)"),
                        use_container_width=True)
        fig = go.Figure()
        for k, a in enumerate(apps):
            fig.add_scatter(x=df.time, y=df[f"{a}_on"] * 0.8 + k, name=APPLIANCE_NAMES[a],
                            line_shape="hv")
        fig.update_yaxes(tickvals=list(range(len(apps))),
                         ticktext=[APPLIANCE_NAMES[a] for a in apps])
        st.plotly_chart(plot_layout(fig, "ON/OFF pattern (training-based thresholds)", 300, ""),
                        use_container_width=True)
        st.caption(f"Range: {df.time.iloc[0]:%d %b %Y %H:%M} to {df.time.iloc[-1]:%H:%M}. "
                   f"Gaps in the lines are missing readings. Region: {REGION}.")

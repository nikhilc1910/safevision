"""
Streamlit dashboard — Live Monitor, Violation Log, Training Results.

Three screens via sidebar nav. No API layer — pipeline runs directly.
Upload a video on Screen 1, review the event log on Screen 2,
inspect training decisions on Screen 3.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# ensure safevision/ is on sys.path regardless of where streamlit is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datetime import date, datetime

import cv2
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db.store import ViolationStore
from inference.pipeline import Pipeline

# ── paths ──────────────────────────────────────────────────────────────────────

_ROOT      = Path(__file__).resolve().parent.parent  # dashboard/app.py → dashboard/ → safevision/
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(_ROOT / "runs" / "aug-freeze_final.pt")))
DB_PATH       = _ROOT / "safevision.db"
TRAINING_META = _ROOT / "runs" / "training_results.json"
WANDB_CURVES  = _ROOT.parent / "docs" / "wandb_results.png"

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeVision",
    page_icon="🦺",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono&display=swap');

/* ── base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', ui-sans-serif, system-ui !important;
    background: #0a0a0a !important;
    color: #e5e5e5 !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1f1f1f !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #a3a3a3 !important;
    padding: 6px 0 !important;
    transition: color .15s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #f5f5f5 !important; }

/* ── headings ── */
h1,h2,h3 {
    font-family: 'Inter', system-ui !important;
    font-weight: 600 !important;
    color: #f5f5f5 !important;
    letter-spacing: -0.01em !important;
}
h2 { font-size: 20px !important; margin-bottom: 4px !important; }

/* ── buttons → pill style ── */
.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 8px 20px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    height: 36px !important;
    transition: opacity .15s !important;
    box-shadow: none !important;
}
.stButton > button:hover { opacity: .82 !important; }

/* ── file uploader ── */
[data-testid="stFileUploader"] > section {
    background: #111111 !important;
    border: 2px dashed #2a2a2a !important;
    border-radius: 12px !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: #3f3f3f !important;
}

/* ── progress bar ── */
.stProgress > div > div > div { background: #3b82f6 !important; border-radius: 9999px !important; }
.stProgress > div > div      { background: #1f1f1f !important; border-radius: 9999px !important; }

/* ── metrics ── */
[data-testid="metric-container"] {
    background: #111111 !important;
    border: 1px solid #1f1f1f !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricValue"] { color: #f5f5f5 !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #737373 !important; font-size: 12px !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] iframe {
    border: 1px solid #1f1f1f !important;
    border-radius: 8px !important;
}

/* ── info / warning strips ── */
.stInfo    { background: #0c1a2e !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; color: #93c5fd !important; }
.stWarning { background: #1a0e00 !important; border: 1px solid #4d2d00 !important; border-radius: 8px !important; }
.stError   { background: #1a0505 !important; border: 1px solid #7f1d1d !important; border-radius: 8px !important; }

/* ── inline code ── */
code {
    font-family: 'JetBrains Mono', ui-monospace !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 12px !important;
    color: #a5f3fc !important;
}

/* ── dividers ── */
hr { border-color: #1f1f1f !important; margin: 20px 0 !important; }

/* ── caption / small text ── */
.stCaption, small { color: #525252 !important; font-size: 12px !important; }

/* ── select / date ── */
[data-testid="stSelectbox"] > div,
[data-testid="stDateInput"]  > div {
    background: #111111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #e5e5e5 !important;
}

/* ── custom components ── */
.sv-stat-row { display: flex; gap: 12px; margin-top: 16px; }
.sv-stat {
    flex: 1;
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.sv-stat-val { font-size: 28px; font-weight: 600; color: #f5f5f5; line-height: 1; }
.sv-stat-lbl { font-size: 12px; color: #525252; margin-top: 6px; font-weight: 500; letter-spacing: .03em; text-transform: uppercase; }
.sv-stat-red  .sv-stat-val { color: #f87171; }
.sv-stat-blue .sv-stat-val { color: #60a5fa; }

/* ── glow border presets (aurora / ocean / sunset / nature) ── */
.sv-glow-aurora { border-color: #6366f1 !important; box-shadow: 0 0 16px rgba(99,102,241,.28), 0 0 0 1px rgba(99,102,241,.12); }
.sv-glow-ocean  { border-color: #0ea5e9 !important; box-shadow: 0 0 16px rgba(14,165,233,.28), 0 0 0 1px rgba(14,165,233,.12); }
.sv-glow-sunset { border-color: #f97316 !important; box-shadow: 0 0 16px rgba(249,115,22,.28),  0 0 0 1px rgba(249,115,22,.12); }
.sv-glow-nature { border-color: #22c55e !important; box-shadow: 0 0 16px rgba(34,197,94,.28),   0 0 0 1px rgba(34,197,94,.12); }

.sv-alert {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #111111;
    border: 1px solid #1f1f1f;
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    animation: slideIn .2s ease;
}
.sv-alert-amber { border-left-color: #f59e0b; }

@keyframes slideIn {
    from { opacity: 0; transform: translateY(-4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.sv-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .02em;
    text-transform: uppercase;
}
.sv-badge-red   { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.sv-badge-amber { background: #1c1000; color: #fbbf24; border: 1px solid #78350f; }

.sv-conf { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.sv-conf-hi  { color: #4ade80; }
.sv-conf-lo  { color: #fbbf24; }

.sv-time { font-size: 11px; color: #404040; margin-left: auto; }

.sv-violation-banner {
    background: #1a0505;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    animation: pulseRed 2s ease-in-out infinite;
}
.sv-violation-banner h4 { margin: 0 0 2px 0; color: #fca5a5; font-size: 14px; font-weight: 600; }
.sv-violation-banner p  { margin: 0; color: #737373; font-size: 12px; }

@keyframes pulseRed {
    0%,100% { border-left-color: #ef4444; }
    50%      { border-left-color: #fca5a5; }
}

.sv-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #404040;
    margin-bottom: 10px;
}

.sv-empty {
    text-align: center;
    padding: 32px 16px;
    color: #404040;
    font-size: 13px;
    border: 1px dashed #1f1f1f;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── shared resources ───────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline(weights_override: str | None = None):
    path = Path(weights_override) if weights_override else MODEL_PATH
    if not path.exists():
        return None
    return Pipeline(path, zone_config=[], camera_id="demo")


@st.cache_resource
def load_store():
    return ViolationStore(DB_PATH)


pipe  = load_pipeline(st.session_state.get("uploaded_weights_path"))
store = load_store()

# ── helpers ────────────────────────────────────────────────────────────────────

VTYPE_COLOR = {
    "no_helmet": "red",
    "no_gloves": "amber",
    "no_boots":  "amber",
    "no_goggle": "amber",
}


def sv_badge(vtype: str) -> str:
    c = VTYPE_COLOR.get(vtype, "red")
    return f'<span class="sv-badge sv-badge-{c}">{vtype}</span>'


def sv_conf(conf: float) -> str:
    cls = "sv-conf-hi" if conf >= 0.70 else "sv-conf-lo"
    return f'<span class="sv-conf {cls}">{conf:.2f}</span>'


def sv_alert_card(a: dict) -> str:
    c = VTYPE_COLOR.get(a["type"], "red")
    glow = "sv-glow-sunset" if c == "red" else "sv-glow-aurora"
    return f"""
<div class="sv-alert {glow}{'  sv-alert-amber' if c == 'amber' else ''}">
  {sv_badge(a['type'])}
  <span style="font-size:12px;color:#737373">zone: {a['zone']}</span>
  {sv_conf(a['confidence'])}
  <span class="sv-time">{a['time']}</span>
</div>"""


def sv_stats(violations: int, frames: int, high_conf: int) -> str:
    return f"""
<div class="sv-stat-row">
  <div class="sv-stat sv-stat-red sv-glow-sunset">
    <div class="sv-stat-val">{violations}</div>
    <div class="sv-stat-lbl">Violations</div>
  </div>
  <div class="sv-stat sv-glow-ocean">
    <div class="sv-stat-val" style="color:#60a5fa">{frames}</div>
    <div class="sv-stat-lbl">Frames</div>
  </div>
  <div class="sv-stat sv-glow-nature">
    <div class="sv-stat-val">{high_conf}</div>
    <div class="sv-stat-lbl">High-conf ≥ 0.70</div>
  </div>
</div>"""


# ── session state ──────────────────────────────────────────────────────────────

if "alerts" not in st.session_state:
    st.session_state.alerts = []

# ── sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("SafeVision")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "nav",
    ["📹 Live Monitor", "📋 Violation Log", "📊 Training Results", "⚙ Settings"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("YOLOv8n · aug-freeze checkpoint")
st.sidebar.caption("Conf threshold: 0.4")


# ══════════════════════════════════════════════════════════════════════════════
# Screen 1 — Live Monitor
# ══════════════════════════════════════════════════════════════════════════════

if page == "📹 Live Monitor":
    st.subheader("Live Monitor")

    if pipe is None:
        st.error(
            f"Model weights not found at `{MODEL_PATH}`. "
            "Run training on Kaggle first, then copy `aug-freeze_final.pt` to that path."
        )
        st.stop()

    st.info(
        "Upload a video clip — the pipeline runs PPE detection + zone intrusion check "
        "and returns an annotated output. Zone polygon drawing is planned for v1.1; "
        "this demo uses a full-frame default zone."
    )

    uploaded = st.file_uploader("Video clip", type=["mp4", "avi", "mov"])

    if uploaded:
        with tempfile.TemporaryDirectory() as tmp:
            in_path  = Path(tmp) / "input.mp4"
            out_path = Path(tmp) / "annotated.mp4"
            in_path.write_bytes(uploaded.read())

            cap = cv2.VideoCapture(str(in_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            bar = st.progress(0, text="Running inference...")

            def on_progress(idx, total):
                pct = min(int(idx / max(total, 1) * 100), 100)
                bar.progress(pct, text=f"Frame {idx} / {total}")

            # Reset between uploads — don't carry dwell counters from the previous clip
            pipe.__init__(MODEL_PATH, zone_config=[], camera_id="demo")

            violations = pipe.run_video(in_path, output_path=out_path, progress_cb=on_progress)
            bar.progress(100, text="Done")

            for v in violations:
                store.insert(v)
                st.session_state.alerts.insert(0, {
                    "type":       v.vtype,
                    "zone":       v.zone_id,
                    "confidence": v.confidence,
                    "time":       datetime.utcnow().strftime("%H:%M:%S"),
                })

            # ── video | alert panel ────────────────────────────────────────────
            vcol, acol = st.columns([3, 2])

            with vcol:
                st.markdown('<div class="sv-section-label">Annotated output</div>', unsafe_allow_html=True)
                if out_path.exists() and out_path.stat().st_size > 0:
                    st.video(out_path.read_bytes())
                else:
                    st.warning("No output video written — codec issue. Check that ffmpeg is on PATH.")

                high_conf = sum(1 for v in violations if v.confidence >= 0.70)
                st.markdown(sv_stats(len(violations), total_frames, high_conf), unsafe_allow_html=True)

            with acol:
                hdr, clr = st.columns([3, 1])
                hdr.markdown('<div class="sv-section-label" style="padding-top:6px">Session alerts</div>', unsafe_allow_html=True)
                if clr.button("Clear", key="clr"):
                    st.session_state.alerts = []

                if st.session_state.alerts:
                    # violation summary banner
                    vtypes = list({a["type"] for a in st.session_state.alerts})
                    vlist  = "  ·  ".join(sorted(vtypes))
                    st.markdown(
                        f'<div class="sv-violation-banner">'
                        f'<h4>⚠ Active violations</h4>'
                        f'<p>{vlist}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    cards = "".join(sv_alert_card(a) for a in st.session_state.alerts[:20])
                    st.markdown(cards, unsafe_allow_html=True)
                else:
                    st.markdown('<div class="sv-empty">No alerts this session</div>', unsafe_allow_html=True)

    else:
        # Show history even when no clip is uploaded — useful on repeat visits
        st.markdown("---")
        st.caption("No clip uploaded yet. Recent violations from previous sessions:")
        recent = store.recent(limit=10)
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
        else:
            st.caption("No violations logged yet.")


# ══════════════════════════════════════════════════════════════════════════════
# Screen 2 — Violation Log
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Violation Log":
    st.subheader("Violation Log")

    recent = store.recent(limit=500)

    if not recent:
        st.caption("No violations logged yet. Upload a video on the Live Monitor screen.")
    else:
        df = pd.DataFrame(recent)

        # ── filters ───────────────────────────────────────────────────────────
        fc, zc, tc, dc = st.columns(4)

        cameras = ["All"] + sorted(df["camera_id"].unique().tolist()) if "camera_id" in df.columns else ["All"]
        zones   = ["All"] + sorted(df["zone_id"].unique().tolist()) if "zone_id" in df.columns else ["All"]
        vtypes  = ["All"] + sorted(df["vtype"].unique().tolist()) if "vtype" in df.columns else ["All"]

        cam_sel  = fc.selectbox("Camera", cameras)
        zone_sel = zc.selectbox("Zone", zones)
        type_sel = tc.selectbox("Type", vtypes)
        date_sel = dc.date_input("Date", value=date.today())

        if cam_sel != "All" and "camera_id" in df.columns:
            df = df[df["camera_id"] == cam_sel]
        if zone_sel != "All" and "zone_id" in df.columns:
            df = df[df["zone_id"] == zone_sel]
        if type_sel != "All" and "vtype" in df.columns:
            df = df[df["vtype"] == type_sel]
        # date filter only if timestamp column exists
        if "timestamp_utc" in df.columns:
            df["_date"] = pd.to_datetime(df["timestamp_utc"], errors="coerce").dt.date
            df = df[df["_date"] == date_sel]
            df = df.drop(columns=["_date"])

        st.markdown(f"Showing **{len(df)}** events")

        display_cols = [c for c in [
            "id", "vtype", "camera_id", "zone_id",
            "confidence", "timestamp_utc", "false_positive"
        ] if c in df.columns]

        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # ── export ────────────────────────────────────────────────────────────
        ex_col, _ = st.columns([1, 4])
        ex_col.download_button(
            label="Export CSV",
            data=df.to_csv(index=False),
            file_name=f"violations_{date.today()}.csv",
            mime="text/csv",
        )

        # ── mark false positive ───────────────────────────────────────────────
        # Operator-marked FPs are saved to the DB — intended to feed active learning later
        st.markdown("---")
        st.markdown("**Mark false positive**")
        fp_id = st.number_input("Violation ID", min_value=1, step=1, label_visibility="collapsed")
        if st.button("Mark as false positive"):
            store.mark_false_positive(int(fp_id))
            st.success(f"ID {fp_id} marked as false positive.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Screen 3 — Training Results
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Training Results":
    st.subheader("Training Results")

    # ── model summary strip ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model",    "YOLOv8n")
    c2.metric("Dataset",  "Construction-PPE")
    c3.metric("Epochs",   "100")
    c4.metric("Warmup",   "10 frozen epochs")
    c5.metric("Hardware", "Kaggle T4 ×2")

    st.markdown("---")

    # Load real numbers from training_results.json once Kaggle run finishes.
    # Until then everything shows TBD — honest about state.
    if TRAINING_META.exists():
        meta = json.loads(TRAINING_META.read_text())
    else:
        meta = {}

    def m(key, fallback="TBD"):
        return meta.get(key, fallback)

    # ── ablation table ────────────────────────────────────────────────────────
    st.markdown("**Ablation results**")

    abl = pd.DataFrame([
        {
            "run":               "baseline",
            "aug":               "✗",
            "freeze":            "✗",
            "mAP50":             m("baseline_map50"),
            "mAP50-95":          m("baseline_map"),
            "recall (no-helmet)": m("baseline_recall_no_helmet"),
        },
        {
            "run":               "aug-only",
            "aug":               "✓",
            "freeze":            "✗",
            "mAP50":             m("aug_only_map50"),
            "mAP50-95":          m("aug_only_map"),
            "recall (no-helmet)": m("aug_only_recall_no_helmet"),
        },
        {
            "run":               "freeze-only",
            "aug":               "✗",
            "freeze":            "✓",
            "mAP50":             m("freeze_only_map50"),
            "mAP50-95":          m("freeze_only_map"),
            "recall (no-helmet)": m("freeze_only_recall_no_helmet"),
        },
        {
            "run":               "aug-freeze ★",
            "aug":               "✓",
            "freeze":            "✓",
            "mAP50":             m("aug_freeze_map50"),
            "mAP50-95":          m("aug_freeze_map"),
            "recall (no-helmet)": m("aug_freeze_recall_no_helmet"),
        },
    ])

    st.dataframe(abl, use_container_width=True, hide_index=True)
    st.caption(
        "★ Best run — aug-freeze checkpoint used in production. "
        "TBD cells fill in after Kaggle training completes."
    )

    st.markdown("---")

    # ── curves | per-class recall ─────────────────────────────────────────────
    curve_col, recall_col = st.columns(2)

    with curve_col:
        st.markdown("**Training curves**")
        if WANDB_CURVES.exists():
            st.image(str(WANDB_CURVES), use_column_width=True)
        else:
            st.info(
                "Save a WandB screenshot to `docs/wandb_curves.png` "
                "after training completes."
            )

    with recall_col:
        st.markdown("**Per-class recall — aug-freeze**")

        classes = ["no_helmet", "no_gloves", "no_boots", "no_goggle", "worker"]
        # pull from aug_freeze_per_class_recall in training_results.json
        per_class = meta.get("aug_freeze_per_class_recall", {})
        recalls = [float(per_class.get(c, 0.0)) for c in classes]

        if not meta:
            st.caption("Populate `runs/training_results.json` after the Kaggle evaluate cell runs.")

        # colour: green if gate passed, amber if borderline, red if below gate
        # safety-critical classes (no-helmet, no-vest) gate at 0.80
        bar_colors = []
        for i, c in enumerate(classes):
            v = recalls[i]
            if v == 0.0:
                bar_colors.append("#2D3748")  # placeholder grey — no data yet
            elif v >= 0.80:
                bar_colors.append("#22C55E")
            elif v >= 0.60:
                bar_colors.append("#D97706")
            else:
                bar_colors.append("#E53E3E")

        fig = go.Figure(go.Bar(
            x=recalls,
            y=classes,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.2f}" if v > 0 else "TBD" for v in recalls],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9", size=12),
            xaxis=dict(range=[0, 1.15], gridcolor="#2D3748", title="recall"),
            yaxis=dict(gridcolor="#2D3748"),
            margin=dict(l=0, r=50, t=10, b=30),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green >= 0.80 | Amber 0.60-0.79 | Red < 0.60 | Grey = no data yet. "
            "no-helmet and no-safety-vest are safety-critical (gate: 0.80 recall)."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Screen 4 - Settings
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙ Settings":
    st.subheader("Settings")
    st.caption("Changes apply to the current session only -- not persisted to .env.")

    st.markdown("---")

    # model weights upload
    st.markdown("**Model weights**")
    if MODEL_PATH.exists():
        st.success(f"Weights loaded: `{MODEL_PATH.name}`")
    else:
        st.warning("No weights found. Upload your `.pt` file to enable Live Monitor.")
        weights_file = st.file_uploader("Upload model weights (.pt)", type=["pt"])
        if weights_file is not None:
            upload_dest = Path(tempfile.gettempdir()) / weights_file.name
            upload_dest.write_bytes(weights_file.read())
            st.session_state["uploaded_weights_path"] = str(upload_dest)
            st.success(f"Weights loaded for this session: `{weights_file.name}`")
            st.caption("Refresh Live Monitor — it will use these weights until the session ends.")

    # use uploaded weights path if set
    if "uploaded_weights_path" in st.session_state:
        import builtins
        builtins.__dict__["_OVERRIDE_MODEL_PATH"] = st.session_state["uploaded_weights_path"]

    st.markdown("---")

    # inference
    st.markdown("**Inference**")

    conf = st.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(st.session_state.get("conf_threshold", 0.4)),
        step=0.05,
        help="Lower = more detections, more false positives. 0.4 is the product default.",
    )
    st.session_state["conf_threshold"] = conf
    st.caption(
        f"Current: {conf:.2f}. "
        "Lower for busy floors where missing a violation is worse than a false alert. "
        "Raise if operators are dismissing too many flagged events."
    )

    st.markdown("---")

    # storage
    st.markdown("**Storage**")

    store_snaps = st.toggle(
        "Store frame snapshots",
        value=st.session_state.get("store_snaps", False),
        help="Saves a JPEG crop of each violation frame to the DB. Each crop is ~20-50KB -- grows fast.",
    )
    st.session_state["store_snaps"] = store_snaps

    if store_snaps:
        st.warning(
            "Snapshot storage is on. The violations table will grow quickly -- "
            "watch disk usage on long sessions."
        )
    else:
        st.caption("Snapshots off. Violation records store metadata only.")

    st.markdown("---")

    # feedback
    st.markdown("**Feedback**")
    st.caption("Bug reports, false positive patterns, feature requests — anything useful.")

    feedback_name = st.text_input("Your name (optional)")
    feedback_text = st.text_area("Feedback", placeholder="e.g. zone intrusion fires on shadows near the door...", height=120)

    if st.button("Send Feedback", disabled=not feedback_text.strip()):
        subject = "SafeVision Feedback"
        if feedback_name.strip():
            subject = f"SafeVision Feedback from {feedback_name.strip()}"
        body = feedback_text.strip().replace("\n", "%0A").replace(" ", "%20")
        mailto = f"mailto:nikhil19102004@gmail.com?subject={subject.replace(' ', '%20')}&body={body}"
        st.markdown(
            f'<a href="{mailto}" target="_blank">'
            '<button style="background:#1e40af;color:white;border:none;padding:8px 18px;'
            'border-radius:6px;cursor:pointer;font-size:14px;">Open in email client</button>'
            '</a>',
            unsafe_allow_html=True,
        )
        st.caption("Click the button above to open your email client with the feedback pre-filled.")

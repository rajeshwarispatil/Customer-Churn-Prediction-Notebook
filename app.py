"""Customer churn dashboard.
Put this file in the same folder as customer_churn (.csv, .xls or .xlsx), then run:
    streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer churn", page_icon="📉", layout="wide")

HERE = Path(__file__).parent
FOUND = next((p for d in (HERE, Path.cwd()) for ext in ("csv", "xlsx", "xls")
              for p in [d / f"customer_churn.{ext}"] if p.exists()), None)
POSITIVE = {"1", "yes", "y", "true", "t", "churned", "left"}
KEEP, CHURN = "#2f7d8c", "#e4572e"


# ---------- data ----------
@st.cache_data
def load(src):
    name = str(getattr(src, "name", src)).lower()
    if name.endswith((".xls", ".xlsx")):
        try:
            df = pd.read_excel(src)
        except Exception:  # some ".xls" files are really CSV text
            if hasattr(src, "seek"):
                src.seek(0)
            df = pd.read_csv(src)
    else:
        df = pd.read_csv(src)
    for c in df.columns:  # e.g. TotalCharges stored as text with blanks
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.9:
                df[c] = conv
    return df


def find_churn_col(df):
    for c in df.columns:
        if any(k in c.lower() for k in ("churn", "exited", "attrition")):
            return c
    return None


def split_columns(df, target):
    ids, cats, nums = [], [], []
    for c in df.columns:
        if c in (target, "_churn"):
            continue
        n = df[c].nunique()
        if n >= 0.9 * len(df) and n > 12:
            ids.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]) and n > 12:
            nums.append(c)
        elif 2 <= n <= 12:
            cats.append(c)
    return ids, cats, nums


if FOUND:
    df = load(FOUND)
else:
    up = st.file_uploader("customer_churn file not found next to app.py. Upload it here:", type=["csv", "xls", "xlsx"])
    if up is None:
        st.stop()
    df = load(up)

churn_col = find_churn_col(df)
if churn_col is None:
    st.error("No churn column found. Rename the target column to 'Churn', 'Exited' or 'Attrition'.")
    st.stop()

df = df.copy()
_num = pd.to_numeric(df[churn_col], errors="coerce")
if _num.notna().mean() > 0.9:  # 0/1 (or 0.0/1.0) numeric target
    df["_churn"] = (_num.fillna(0) > 0).astype(int)
else:
    df["_churn"] = df[churn_col].astype(str).str.strip().str.lower().isin(POSITIVE).astype(int)
id_cols, cat_cols, num_cols = split_columns(df, churn_col)

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
view = df
for c in cat_cols[:5]:
    opts = sorted(df[c].dropna().astype(str).unique())
    pick = st.sidebar.multiselect(c, opts, default=opts)
    view = view[view[c].astype(str).isin(pick)]
st.sidebar.caption(f"{len(view):,} of {len(df):,} customers selected")

# ---------- header + KPIs ----------
st.title("Customer churn")
n, churned = len(view), int(view["_churn"].sum())
rate = churned / n if n else 0
k = st.columns(4)
k[0].metric("Churn rate", f"{rate:.1%}")
k[1].metric("Customers", f"{n:,}")
k[2].metric("Churned", f"{churned:,}")
k[3].metric("Retained", f"{n - churned:,}")

tab1, tab2, tab3 = st.tabs(["Overview", "Drivers", "Predict"])

# ---------- overview ----------
with tab1:
    a, b = st.columns(2)
    with a:
        col = st.selectbox("Churn rate by", cat_cols, key="grp")
        g = view.groupby(col)["_churn"].agg(["mean", "count"]).reset_index()
        g["mean"] *= 100
        fig = px.bar(g.sort_values("mean"), x="mean", y=col, orientation="h",
                     hover_data={"count": True}, labels={"mean": "Churn rate (%)"})
        fig.update_traces(marker_color=CHURN)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        if num_cols:
            col = st.selectbox("Distribution of", num_cols, key="num")
            plot = view.assign(Status=view["_churn"].map({1: "Churned", 0: "Retained"}))
            fig = px.histogram(plot, x=col, color="Status", nbins=30, barmode="stack",
                               color_discrete_map={"Churned": CHURN, "Retained": KEEP})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns to plot.")

    st.subheader("Groups with the highest churn")
    rows = []
    for c in cat_cols:
        s = view.groupby(c)["_churn"].agg(["mean", "count"])
        for val, r in s.iterrows():
            if r["count"] >= 30:
                rows.append({"Column": c, "Value": str(val), "Customers": int(r["count"]), "Churn rate": r["mean"] * 100})
    if rows:
        top = pd.DataFrame(rows).sort_values("Churn rate", ascending=False).head(10)
        st.dataframe(top, hide_index=True, use_container_width=True,
                     column_config={"Churn rate": st.column_config.ProgressColumn(
                         "Churn rate", format="%.1f%%", min_value=0, max_value=100)})

# ---------- model ----------
features = cat_cols + num_cols


@st.cache_resource
def train(data, features, cat_cols, num_cols):
    cat_cols, num_cols = list(cat_cols), list(num_cols)
    X = pd.get_dummies(data[features].assign(**{c: data[c].astype(str) for c in cat_cols}), columns=cat_cols)
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    y = data["_churn"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    return model, list(X.columns), medians, auc


def encode(rows):
    X = pd.get_dummies(rows[features].assign(**{c: rows[c].astype(str) for c in cat_cols}), columns=cat_cols)
    return X.reindex(columns=cols, fill_value=0).fillna(medians)


model, cols, medians, auc = train(df, features, tuple(cat_cols), tuple(num_cols))
imp = pd.Series(model.feature_importances_, index=cols)


def base_feature(name):  # map dummy column back to its source column
    for c in cat_cols:
        if name.startswith(c + "_"):
            return c
    return name


by_feature = imp.groupby(imp.index.map(base_feature)).sum().sort_values(ascending=False)

with tab2:
    st.caption(f"Random forest, hold-out ROC AUC: {auc:.3f}")
    fig = px.bar(by_feature.head(12).sort_values().reset_index(), x=0, y="index", orientation="h",
                 labels={"0": "Importance", "index": ""})
    fig.update_traces(marker_color=KEEP)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Score one customer")
        top_feats = list(by_feature.head(8).index)
        row = {}
        for f in features:
            if f in top_feats:
                if f in cat_cols:
                    row[f] = st.selectbox(f, sorted(df[f].dropna().astype(str).unique()), key="p_" + f)
                else:
                    row[f] = st.number_input(f, value=float(df[f].median()), key="p_" + f)
            else:
                row[f] = df[f].mode()[0] if f in cat_cols else df[f].median()
        p = model.predict_proba(encode(pd.DataFrame([row])))[0, 1]
        st.metric("Churn probability", f"{p:.1%}")
        st.progress(float(p))
    with right:
        st.subheader("Customers most at risk")
        scored = view.copy()
        scored["Churn probability"] = model.predict_proba(encode(scored))[:, 1] * 100
        show = (id_cols[:1] + ["Churn probability", churn_col] + features[:4])
        risky = scored[scored["_churn"] == 0].sort_values("Churn probability", ascending=False)[show].head(25)
        st.dataframe(risky, hide_index=True, use_container_width=True,
                     column_config={"Churn probability": st.column_config.ProgressColumn(
                         "Churn probability", format="%.0f%%", min_value=0, max_value=100)})
        st.download_button("Download at-risk list (CSV)", risky.to_csv(index=False), "at_risk_customers.csv")
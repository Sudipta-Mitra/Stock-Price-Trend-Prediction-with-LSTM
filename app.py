
import io
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scaler Playground + Quick EDA", page_icon="🧪", layout="wide")

st.title("🧪 Scaler Playground + 📊 Quick EDA")
st.caption("A handy Streamlit app to explore datasets and apply a saved scaler (e.g., StandardScaler/MinMaxScaler).")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Scaler Playground", "CSV EDA", "About"])

    st.markdown("---")
    st.subheader("Theme")
    theme = st.selectbox("Accent", ["Default", "Minimal"])
    if theme == "Minimal":
        st.write("Using a minimal look — best paired with wide layout.")

def load_pickle(file_bytes) -> object:
    try:
        return pickle.loads(file_bytes)
    except Exception as e:
        st.error(f"Could not load pickle: {e}")
        return None

def infer_feature_count_from_scaler(scaler) -> Optional[int]:
    # Try best-effort to infer number of expected features
    for attr in ["n_features_in_", "n_features_in", "n_features_"]:
        if hasattr(scaler, attr):
            try:
                val = int(getattr(scaler, attr))
                if val > 0:
                    return val
            except Exception:
                pass
    if hasattr(scaler, "feature_names_in_"):
        try:
            names = list(getattr(scaler, "feature_names_in_"))
            return len(names)
        except Exception:
            pass
    return None

def get_feature_names(scaler, fallback_n: int) -> List[str]:
    if hasattr(scaler, "feature_names_in_"):
        try:
            return list(getattr(scaler, "feature_names_in_"))
        except Exception:
            pass
    # fallback generic names
    return [f"f{i+1}" for i in range(fallback_n)]

if page == "Scaler Playground":
    st.header("Scaler Playground")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.subheader("Upload a pre-fitted scaler (.pkl)")
        scaler_file = st.file_uploader("Choose scaler.pkl", type=["pkl", "pickle"], accept_multiple_files=False)
        scaler = None
        if scaler_file is not None:
            scaler = load_pickle(scaler_file.read())

    with col_up2:
        st.subheader("Or try loading from app folder (advanced)")
        path = st.text_input("Local path (server-side):", value="scaler.pkl")
        if st.button("Load from path"):
            try:
                with open(path, "rb") as f:
                    scaler = pickle.load(f)
                st.success(f"Loaded scaler from '{path}'")
            except FileNotFoundError:
                st.warning("File not found at the given path.")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    if scaler is None:
        st.info("⬆️ Upload or load a scaler to continue.")
    else:
        with st.expander("ℹ️ Scaler details", expanded=True):
            st.write(f"**Type:** `{type(scaler).__name__}`")
            nfeat = infer_feature_count_from_scaler(scaler)
            if hasattr(scaler, "feature_names_in_"):
                st.write("**feature_names_in_:**", list(getattr(scaler, "feature_names_in_")))
            st.write(f"**Expected #features:** {nfeat if nfeat else 'Unknown'}")

        tab1, tab2 = st.tabs(["🔢 Single-row transform", "📂 Batch CSV transform"])

        with tab1:
            st.subheader("Enter feature values")
            if nfeat is None:
                nfeat = st.number_input("How many features does the scaler expect?", min_value=1, max_value=200, value=3, step=1)
            feat_names = get_feature_names(scaler, nfeat)
            cols = st.columns(min(4, len(feat_names)) or 1)
            inputs = []
            for i, name in enumerate(feat_names):
                with cols[i % len(cols)]:
                    val = st.number_input(f"{name}", value=0.0, format="%.6f", key=f"single_{i}")
                    inputs.append(val)
            arr = np.array(inputs, dtype=float).reshape(1, -1)

            if st.button("Transform ▶️"):
                try:
                    transformed = scaler.transform(arr)
                    st.success("Transformed vector:")
                    st.write(pd.DataFrame(transformed, columns=feat_names))
                except Exception as e:
                    st.error(f"Transform failed: {e}")

        with tab2:
            st.subheader("Upload CSV for batch transform")
            csv_file = st.file_uploader("CSV file", type=["csv"], key="csv_batch")
            if csv_file is not None:
                try:
                    df = pd.read_csv(csv_file)
                    st.write("Preview:", df.head())
                    use_cols_mode = st.radio("Select columns to transform", ["Use all numeric columns", "Pick specific columns"])
                    if use_cols_mode == "Pick specific columns":
                        candidates = list(df.columns)
                        cols_to_use = st.multiselect("Columns", candidates)
                    else:
                        cols_to_use = df.select_dtypes(include=[np.number]).columns.tolist()

                    st.caption(f"Columns to transform: {cols_to_use}")

                    method = st.radio("Method", ["Use pre-fitted scaler.transform", "Fit on uploaded data (scaler.fit_transform)"])

                    if st.button("Run batch transform ▶️"):
                        work = df.copy()
                        if len(cols_to_use) == 0:
                            st.warning("No columns selected.")
                        else:
                            X = work[cols_to_use].to_numpy()
                            try:
                                if method == "Use pre-fitted scaler.transform":
                                    X_scaled = scaler.transform(X)
                                else:
                                    scaler.fit(X)
                                    X_scaled = scaler.transform(X)
                                work[[f"{c}_scaled" for c in cols_to_use]] = X_scaled
                                st.success("Done. Preview of transformed data:")
                                st.dataframe(work.head())

                                buf = io.BytesIO()
                                work.to_csv(buf, index=False)
                                st.download_button("⬇️ Download transformed CSV", buf.getvalue(), file_name="transformed.csv", mime="text/csv")
                            except Exception as e:
                                st.error(f"Batch transform failed: {e}")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")

elif page == "CSV EDA":
    st.header("Quick EDA")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="csv_eda")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.subheader("Dataset overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Numeric cols", f"{df.select_dtypes(include=[np.number]).shape[1]:,}")
        c4.metric("Missing cells", f"{df.isna().sum().sum():,}")

        st.subheader("Peek")
        st.dataframe(df.head())

        st.subheader("Summary (numeric)")
        st.dataframe(df.describe())

        with st.expander("Missing values by column"):
            miss = df.isna().sum().sort_values(ascending=False)
            miss_df = miss[miss > 0].to_frame("missing_count")
            if miss_df.empty:
                st.info("No missing values found 🎉")
            else:
                st.dataframe(miss_df)

        with st.expander("Correlation heatmap (numeric)"):
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 2:
                st.info("Need at least 2 numeric columns for a heatmap.")
            else:
                corr = num.corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(corr.values, aspect='auto')
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns)
                ax.set_title("Correlation")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)

else:
    st.header("About")
    st.markdown(
        """
        **What is this?**  
        A simple Streamlit app to:
        - Upload and use a saved scaler (`.pkl`) for single-row or batch transforms.
        - Do a quick EDA on a CSV (shape, head, describe, missing, and a correlation heatmap).

        **How to run locally**
        ```bash
        pip install streamlit pandas numpy matplotlib scikit-learn
        streamlit run app.py
        ```

        **Notes**
        - Your scaler should be *pre-fitted* for `transform` to work.
        - If your scaler has `feature_names_in_`, the app uses these to build the input form.
        - Batch mode lets you choose between using the pre-fit scaler or fitting on your uploaded data.
        """
    )

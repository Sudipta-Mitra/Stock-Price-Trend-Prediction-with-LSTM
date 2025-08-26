import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scaler Playground + Quick EDA + Time Series", page_icon="🧪", layout="wide")

st.title("Stock-Price-Trend-Prediction-with-LSTM")
st.caption("Upload a fitted scaler (.joblib), explore datasets, and visualize trends by date/year.")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Scaler Playground", "CSV EDA", "📅 Time Series Visualization", "About"])


# --- Helpers ---
def infer_feature_count(scaler):
    for attr in ["n_features_in_", "n_features_in", "n_features_"]:
        if hasattr(scaler, attr):
            return int(getattr(scaler, attr))
    if hasattr(scaler, "feature_names_in_"):
        return len(scaler.feature_names_in_)
    return None

def get_feature_names(scaler, n):
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return [f"f{i+1}" for i in range(n)]


# --- Pages ---
if page == "Scaler Playground":
    st.header("Scaler Playground")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload a pre-fitted scaler (.joblib)")
        scaler_file = st.file_uploader("Choose scaler.joblib", type=["joblib"])
        scaler = None
        if scaler_file:
            try:
                scaler = joblib.load(scaler_file)
                st.success("Scaler loaded successfully ✅")
            except Exception as e:
                st.error(f"Failed to load scaler: {e}")

    with col2:
        st.subheader("Or load from local path (server-side)")
        path = st.text_input("Local path:", value="scaler.joblib")
        if st.button("Load from path"):
            try:
                scaler = joblib.load(path)
                st.success(f"Loaded scaler from '{path}'")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    if scaler is None:
        st.info("⬆️ Upload or load a scaler to continue.")
    else:
        st.write(f"**Scaler type:** {type(scaler).__name__}")
        nfeat = infer_feature_count(scaler)
        feat_names = get_feature_names(scaler, nfeat or 3)
        st.write(f"**Expected features:** {nfeat if nfeat else 'Unknown'}")

        tab1, tab2 = st.tabs(["🔢 Single-row transform", "📂 Batch CSV transform"])

        with tab1:
            st.subheader("Enter feature values")
            if nfeat is None:
                nfeat = st.number_input("Number of features?", min_value=1, max_value=50, value=3, step=1)
                feat_names = get_feature_names(scaler, nfeat)

            cols = st.columns(min(4, len(feat_names)))
            values = []
            for i, name in enumerate(feat_names):
                with cols[i % len(cols)]:
                    values.append(st.number_input(name, value=0.0, format="%.4f", key=f"feat_{i}"))

            arr = np.array(values).reshape(1, -1)

            if st.button("Transform ▶️"):
                try:
                    transformed = scaler.transform(arr)
                    st.write("**Transformed output:**")
                    st.dataframe(pd.DataFrame(transformed, columns=feat_names))
                except Exception as e:
                    st.error(f"Transform failed: {e}")

        with tab2:
            st.subheader("Upload CSV for batch transform")
            csv_file = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
            if csv_file:
                df = pd.read_csv(csv_file)
                st.write("Preview:", df.head())

                cols_to_use = st.multiselect("Columns to transform", df.columns.tolist(),
                                             default=df.select_dtypes(include=[np.number]).columns.tolist())

                if st.button("Run batch transform ▶️"):
                    try:
                        X = df[cols_to_use].to_numpy()
                        X_scaled = scaler.transform(X)
                        df[[f"{c}_scaled" for c in cols_to_use]] = X_scaled
                        st.success("Transformation complete")
                        st.dataframe(df.head())

                        buf = io.BytesIO()
                        df.to_csv(buf, index=False)
                        st.download_button("⬇️ Download transformed CSV", buf.getvalue(),
                                           file_name="transformed.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Batch transform failed: {e}")


elif page == "CSV EDA":
    st.header("Quick EDA")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="eda_csv")
    if csv_file:
        df = pd.read_csv(csv_file)
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.write("Preview:", df.head())
        st.write("Summary:", df.describe())
        if st.checkbox("Show correlation heatmap"):
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] >= 2:
                corr = num.corr()
                fig, ax = plt.subplots()
                im = ax.imshow(corr.values, cmap="coolwarm")
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr.columns)))
                ax.set_yticklabels(corr.columns)
                fig.colorbar(im)
                st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation heatmap.")


elif page == " Time Series Visualization":
    st.header(" Time Series Visualization")

    file = st.file_uploader("Upload a CSV with a Date column", type=["csv"], key="ts_csv")
    if file:
        try:
            df = pd.read_csv(file, parse_dates=["Date"])
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.write("Preview:", df.head())

        # Extract year & month
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.to_period("M")

        # Sidebar year filter
        years = sorted(df["Year"].unique())
        selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

        df_filtered = df[df["Year"].isin(selected_years)]

        # Choose column to plot
        numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            col_to_plot = st.selectbox("Select numeric column to visualize", numeric_cols)

            # Monthly trend
            trend = df_filtered.groupby("Month")[col_to_plot].mean()

            st.subheader("Monthly Trend")
            st.line_chart(trend)

            # Yearly summary
            yearly = df_filtered.groupby("Year")[col_to_plot].mean()
            st.subheader("Yearly Summary")
            st.bar_chart(yearly)
        else:
            st.warning("No numeric columns found to visualize.")


else:
    st.header("About")
    st.markdown("""
    This app lets you:
    - Upload a saved scaler (`.joblib`) and transform new data.
    - Upload a CSV for quick exploratory analysis.
    -  Visualize time series data by date and year.
    """)

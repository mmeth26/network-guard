import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

@st.cache_data(max_entries=2, show_spinner="Running saved detectors...")
def calculate_scores(measurements, _bundle, model_version):
    isolation_data = _bundle["if_preprocess"].transform(measurements)
    forest_data = _bundle["rf_preprocess"].transform(measurements)
    forest = _bundle["random_forest"]
    attack_class_index = list(forest.classes_).index(1)

    scores = {
        "random_forest": forest.predict_proba(forest_data)[:, attack_class_index],
        "isolation_forest": (-_bundle["isolation_forest"].score_samples(isolation_data)),
        "simple_rule": np.abs(isolation_data).max(axis=1)
    }
    return scores

st.set_page_config(
    page_title="Network Guard", 
    page_icon="🛡️", 
    layout="wide"
)

st.title("Network Guard")
st.write("Upload network-flow data and explore model detections.")

st.info(
    "Experimental project. Alerts indicate flows to review, "
    "not confirmed malicious activity."
)

input_mode = st.radio(
    "Choose input type",
    ["View scored results", "Analyze raw flow CSV"]
)

uploaded_file = st.file_uploader(
    "Upload network-flow data (CSV)", 
    type=["csv"]
)

if uploaded_file is not None:
    st.success(f"Selected file: {uploaded_file.name}")


    try:
        df = pd.read_csv(uploaded_file, nrows=50000)
        df.columns = df.columns.str.strip()  # Strip whitespace from column names
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        st.error(f"Could not read this CSV: {e}")
        st.stop()

    if df.empty:
        st.error("This CSV contains no data rows.")
        st.stop()

    if df.columns.duplicated().any():
        st.error("This CSV contains duplicate colum names.")
        st.stop()
    
    st.caption("Analyzing up to the first 50,000 rows. This is not a random sample of the full file.")

    if input_mode == "Analyze raw flow CSV":
        try:
            # Load only the model trained locally 
            model_path = Path("v3_grouped_run1/models.joblib")
            bundle = joblib.load(model_path)

            features = bundle["features"]
            missing_columns = [
                col for col in features if col not in df.columns
            ]

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.stop()

            #Prepare measurements using the same cleaning as training 
            measurements = df[features].apply(
                pd.to_numeric, errors="coerce"
            )

            measurements = measurements.replace(
                [np.inf, -np.inf], np.nan
            )

            measurements = measurements.mask(measurements < 0)

            if measurements.isna().all(axis=1).any():
                st.error("Some rows have no usable measurements.")
                st.stop()

            missing_cells = int(measurements.isna().sum().sum())

            if missing_cells:
                st.warning(
                    f"{missing_cells:,} missing or invalid measurement values "
                    "will used the saved training imputation values."
                )

            #Transform only. Do not retain processing.
            model_version = model_path.stat().st_mtime_ns
            scores = calculate_scores(measurements, bundle, model_version)

            for name, values, in scores.items():
                df[f"{name}_score"] = values
                df[f"{name}_alert"] = (values > bundle["thresholds"][name])
            st.success("Saved models finished analyzing the flows.")

        except FileNotFoundError:
            st.error(
                "Model files not found. Check that "
                "v3_grouped_run1/models.joblib " 
                "is inside your project folder."
            )
            st.stop()

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

    required_columns = [
        f"{name}_{suffix}"
        for name in ["random_forest", "isolation_forest", "simple_rule"]
        for suffix in ["score", "alert"]
    ]

    if any(col not in df.columns for col in required_columns):
        st.error(
            "This file has no complete detector results. "
            "Choose 'Analyze raw flow CSV' to generate them."
        )
        st.stop()

    for name in ["random_forest", "isolation_forest", "simple_rule"]:
        score_col = f"{name}_score"
        alert_col = f"{name}_alert"

        numeric_scores = pd.to_numeric(df[score_col], errors="coerce")
        if not np.isfinite(numeric_scores).all():
            st.error(f"{score_col} contains missing or invalid scores.")
            st.stop()

        alert_text = df[alert_col].astype(str).str.strip().str.lower()
        if not alert_text.isin(["true", "false"]).all():
            st.error(f"{alert_col} must contain True or False on every row.")
            st.stop()

        df[score_col] = numeric_scores
        df[alert_col] = alert_text.eq("true")

    st.write(f"Loaded {len(df):,} network flows.")
    st.dataframe(df.head(100))    

    detector = st.selectbox( 
        "Choose a detection model",
        ["random_forest", "isolation_forest", "simple_rule"]
    )
    st.write(f"Selected model: {detector}")

    alert_column = f"{detector}_alert"
    alerts = df[alert_column]

    flagged_count = int(alerts.sum())
    st.metric("Flows flagged for review", flagged_count)

    score_column = f"{detector}_score"
    flagged_flows = df[alerts].sort_values(
        by=score_column, 
        ascending=False
    )

    if "Label" in df.columns:
        selected_labels = st.multiselect(
            "Filter flagged flows by traffic label",
            options=sorted(df["Label"].dropna().unique()),
        )

        if selected_labels:
            flagged_flows = flagged_flows[
                flagged_flows["Label"].isin(selected_labels)
            ]

    st.write(f"{len(flagged_flows):,} flows flagged match your filters.")

    if st.checkbox("Show flagged flows", value=True):
        st.subheader("Flagged Flows")
        st.dataframe(flagged_flows.head(100))

    st.download_button(
        label="Download flagged flows",
        data=flagged_flows.to_csv(index=False),
        file_name=f"{detector}_flagged_flows.csv",
        mime="text/csv"
    )

    st.subheader("Detection Performance")

    if "Label" in df.columns:
        label = df["Label"].astype("string").str.strip()
        valid_labels = label.notna() & label.ne("")

        if valid_labels.all():
            actual_attack = label.str.upper().ne("BENIGN")

            true_positives = int((alerts & actual_attack).sum())
            false_positives = int((alerts & ~actual_attack).sum())
            false_negatives = int((~alerts & actual_attack).sum())

            total_alerts = true_positives + false_positives
            total_attacks = true_positives + false_negatives

            precision = (
                true_positives / total_alerts 
                if total_alerts > 0 else None
            )

            recall = (
                true_positives / total_attacks 
                if total_attacks > 0 else None
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Attack flows flagged", true_positives)
            col2.metric("False alerts", false_positives)
            col3.metric("Attack flows missed", false_negatives)

            col4, col5 = st.columns(2)

            col4.metric(
                "Precision",
                f"{precision:.1%}" if precision is not None else "N/A"
            )
            
            col5.metric(
                "Recall",
                f"{recall:.1%}" if recall is not None else "N/A"
            )

            st.caption(
                "Performance uses the CSV reference labels. "
                "Metrics cover all loaded rows."
            )
        else:
            st.warning(
                "Some labels are missing or empty. Performance was not calculated."
            )
    else:
        st.info(
            "This CSV has no Label column, so performance metrics cannot be calculated."
        )

    st.subheader("Detector Comparison")
    comparison_rows = []

    for name in ["random_forest", "isolation_forest", "simple_rule"]:
        detector_alerts = df[f"{name}_alert"]
        flagged = int(detector_alerts.sum())
        comparison_rows.append({
            "Detector": name,
            "Flagged Flows": flagged,
            "Flagged percentage": (
                f"{flagged / len(df):.2%}" if len(df) else "N/A"
            )
        })

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, hide_index=True)
    st.caption(
        "More flagged flows does not necessarily mean better detection. "
    )

    st.subheader("Score Distribution")

    scores = pd.to_numeric(df[f"{detector}_score"], errors="coerce")
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()

    if not scores.empty:
        count, edges = np.histogram(scores, bins=20)
        scores_ranges = [f"{edges[i]:.3f} to {edges[i+1]:.3f}" for i in range(len(count))]
        distribution = pd.DataFrame({
            "Score Range": scores_ranges,
            "Number of Flows": count
        })
        st.bar_chart(distribution.set_index("Score Range")
        )
    else:
        st.info("No valid scores available.")
    st.caption(
        "Each bar counts flows within a score range. "
        "Higher scores indicate stronger review signals "
        "within the selected detector, not attack probabilities. "
    )
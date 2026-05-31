import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

st.set_page_config(
    page_title="Fast India Crime Map", layout="wide", page_icon="../stock/logo.png"
)

project_root = Path(__file__).resolve().parent.parent


# Load saved Models & Files
@st.cache_resource
def load_models_data():
    model_path = project_root / "models"
    svr = joblib.load(model_path / "svr_model.pkl")
    state_enc = joblib.load(model_path / "state_encoder.pkl")
    crime_enc = joblib.load(model_path / "crime_encoder.pkl")

    csv_loc = project_root / "data" / "processed" / "crime_data_processed.csv"
    df = pd.read_csv(csv_loc)

    # Ensuring Field is numeric
    df["Crime Count"] = pd.to_numeric(df["Crime Count"], errors="coerce").fillna(0)

    history_df = df[df["Year"] <= 2023].copy()
    history_df = history_df.sort_values(["State", "Crime Description", "Year", "Month"])

    geojson_loc = project_root / "json" / "cleaned.geojson"
    with open(geojson_loc, "r") as f:
        india_states = json.load(f)

    all_states = [
        feature["properties"]["st_nm"] for feature in india_states["features"]
    ]
    all_states_df = pd.DataFrame({"State": all_states})
    return svr, state_enc, crime_enc, history_df, df, india_states, all_states_df


# ============================
try:
    svr, state_enc, crime_enc, history_df_base, df, india_states, all_states_df = (
        load_models_data()
    )
except:
    st.error("Models not found. Make sure you ran the training script!")
    st.stop()
# ============================
# Ensure cache with data
if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = history_df_base.copy()
    st.session_state.last_generated_year = 2023
# ============================


def visualization():
    st.title("India Crime Data Visualization")

    # Define the input fields
    with st.form("filters_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            year_filter = st.selectbox(
                "Year", sorted(df[df["Year"] < 2024]["Year"].unique()), index=0
            )

        with col2:
            months = sorted(df["Month"].unique())
            month_filter = st.selectbox(
                "Month", ["All"] + [str(m) for m in months], index=0
            )
        with col3:
            crimes = sorted(df["Crime Description"].unique())
            crime_filter = st.selectbox("Crime Type", ["All"] + crimes, index=0)
        submitted = st.form_submit_button("Update Map")

    df["Crime Count"] = pd.to_numeric(df["Crime Count"], errors="coerce").fillna(0)

    # OnSubmit Filter data from Dataset
    if submitted:
        df_filtered = df[df["Year"] == year_filter]

        if month_filter != "All":
            df_filtered = df_filtered[df_filtered["Month"] == int(month_filter)]

        if crime_filter != "All":
            df_filtered = df_filtered[df_filtered["Crime Description"] == crime_filter]

        df_agg = (
            df_filtered.groupby("State", as_index=False)["Crime Count"].sum().fillna(0)
        )

        merged_df = all_states_df.merge(df_agg, on="State", how="left").fillna(0)
        merged_df["Crime Count"] = merged_df["Crime Count"].astype(int)

        title = f"Crime Data for {year_filter}"
        if month_filter != "All":
            title += f" - Month {month_filter}"
        if crime_filter != "All":
            title += f" ({crime_filter})"
        else:
            title += " (All Crimes)"

        fig = px.choropleth(
            merged_df,
            geojson=india_states,
            locations="State",
            featureidkey="properties.st_nm",
            color="Crime Count",
            color_continuous_scale="Reds",
            title=title,
        )
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="#AEDBFE")
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=650)

        st.plotly_chart(fig, width="stretch")
        st.success("Map updated ")
    else:
        st.info("Select filters and click **Update Map** to load visualization.")


# ============================
#   Prediction / Forecasting Generator
def ensure_forecast_exists(target_year):
    """
    Since SVR treats time series as a regression problem, it requires context from the recent past
    to understand the current trend. Lags provide 'short-term memory', allowing
    the model to predict the next value based on immediate history rather than just the general monthly average.
    """
    current_max = st.session_state.last_generated_year
    if target_year <= current_max:
        return
    years_to_gen = range(current_max + 1, target_year + 1)
    progress_text = "Forecasting timelines... Please wait."
    my_bar = st.progress(0, text=progress_text)

    cache = st.session_state.forecast_cache.copy()
    unique_combos = history_df_base[["State", "Crime Description"]].drop_duplicates()
    total_steps = len(years_to_gen) * 12
    current_step = 0
    # Generate a year more than the selected, for faster access of data if needed.
    for year in years_to_gen:
        for month in range(1, 13):
            current_step += 1
            progress = current_step / total_steps
            my_bar.progress(progress, text=progress_text)

            this_month_input = unique_combos.copy()
            this_month_input["Year"] = year
            this_month_input["Month"] = month
            this_month_input["Crime Count"] = 0

            temp_df = pd.concat([cache, this_month_input], ignore_index=True)

            # Force numeric type to prevent DataError in rolling mean
            temp_df["Crime Count"] = pd.to_numeric(
                temp_df["Crime Count"], errors="coerce"
            ).fillna(0)

            temp_df = temp_df.sort_values(
                ["State", "Crime Description", "Year", "Month"]
            )

            grouper = temp_df.groupby(["State", "Crime Description"])["Crime Count"]

            temp_df["Lag_1"] = grouper.shift(1)
            temp_df["Lag_3"] = grouper.shift(3)
            temp_df["Rolling_Mean_3"] = grouper.transform(
                lambda x: x.shift(1).rolling(3).mean()
            )
            temp_df["Month_sin"] = np.sin(2 * np.pi * temp_df["Month"] / 12)
            temp_df["Month_cos"] = np.cos(2 * np.pi * temp_df["Month"] / 12)
            X_pred = temp_df[
                (temp_df["Year"] == year) & (temp_df["Month"] == month)
            ].copy()

            features = [
                "State",
                "Year",
                "Crime Description",
                "Lag_1",
                "Lag_3",
                "Rolling_Mean_3",
                "Month_sin",
                "Month_cos",
            ]
            X_pred[features] = X_pred[features].fillna(0)

            try:

                preds = svr.predict(X_pred[features])
                preds = [max(0, int(round(p))) for p in preds]
                X_pred["Crime Count"] = preds
                cols_to_keep = [
                    "State",
                    "Crime Description",
                    "Year",
                    "Month",
                    "Crime Count",
                ]
                # Storing the data to cache.
                cache = pd.concat([cache, X_pred[cols_to_keep]], ignore_index=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.stop()

    my_bar.empty()
    st.session_state.forecast_cache = cache
    st.session_state.last_generated_year = target_year
    # st.success(f"Forecast for {target_year - 1}  generated successfully")
    st.success(f"Forecast generated successfully")


# ==================================


# Get Next 3,6, 12 month prediction from current selection
def get_long_term_forecast_from_cache(state, start_year, start_month, crime_type):
    end_year = start_year + 1
    ensure_forecast_exists(end_year)

    cache = st.session_state.forecast_cache

    mask = cache["State"] == state
    if crime_type != "All Crimes":
        mask &= cache["Crime Description"] == crime_type

    df_state = cache[mask].copy()

    df_state["Date"] = pd.to_datetime(df_state[["Year", "Month"]].assign(Day=1))
    start_date = pd.to_datetime(f"{start_year}-{start_month}-01")

    end_q = start_date + pd.DateOffset(months=3)
    q_val = df_state[(df_state["Date"] >= start_date) & (df_state["Date"] < end_q)][
        "Crime Count"
    ].sum()

    end_h = start_date + pd.DateOffset(months=6)
    h_val = df_state[(df_state["Date"] >= start_date) & (df_state["Date"] < end_h)][
        "Crime Count"
    ].sum()

    end_y = start_date + pd.DateOffset(months=12)
    y_val = df_state[(df_state["Date"] >= start_date) & (df_state["Date"] < end_y)][
        "Crime Count"
    ].sum()

    return int(q_val), int(h_val), int(y_val)


# Pre
all_crime_types = list(crime_enc.classes_)
all_crime_types.insert(0, "All Crimes")


def show_crime_prediction():
    st.title("India Crime Data Forecasting")
    if "selected_state" not in st.session_state:
        st.session_state.selected_state = state_enc.classes_[0]

    st.markdown(
        f'<p style="font-size:18px; font-weight:bold;">Selected State: <span style="color:#4B8BBE;">{st.session_state.selected_state}</span></p>',
        unsafe_allow_html=True,
    )
    tab1, tab2 = st.tabs(["Prediction Dashboard", "Heatmap Analysis"])

    selected_state = None

    # ==== 1: PREDICTION DASHBOARD =====
    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Select Parameters")

            all_states = state_enc.classes_
            selected_state = st.selectbox(
                "Select State",
                all_states,
                index=all_states.tolist().index(st.session_state.selected_state),
                key="selected_state",  # Syncs to session state
            )

            col_y, col_m = st.columns(2)
            selected_year = col_y.number_input(
                "Year", min_value=2024, max_value=2030, value=2025
            )
            selected_month = col_m.selectbox(
                "Month",
                range(1, 13),
                index=0,  # Default to January
                format_func=lambda x: pd.to_datetime(f"2022-{x}-01").strftime("%B"),
            )

            selected_crime = st.selectbox("Crime Type", all_crime_types, index=0)

            run_pred = st.button("Generate Prediction", type="primary")

        with col2:
            if run_pred:
                st.session_state.auto_update_map = True
                if selected_year >= 2026:
                    st.warning(
                        "**Warning**: Predicting > 2 years ahead relies on recursive estimation and may have uncertainty.",
                        icon="⚠️",
                    )

                ensure_forecast_exists(selected_year + 1)
                cache = st.session_state.forecast_cache

                mask = (
                    (cache["Year"] == selected_year)
                    & (cache["Month"] == selected_month)
                    & (cache["State"] == selected_state)
                )
                if selected_crime != "All Crimes":
                    mask &= cache["Crime Description"] == selected_crime

                df_pred = cache[mask].copy()

                if not df_pred.empty:
                    st.subheader(
                        f"Forecast: {selected_state} ({pd.to_datetime(f'2022-{selected_month}-01').strftime('%B')} {selected_year})"
                    )

                    month_total = df_pred["Crime Count"].sum()
                    if not df_pred.empty:
                        top_crime = df_pred.loc[df_pred["Crime Count"].idxmax()][
                            "Crime Description"
                        ]
                    else:
                        top_crime = "N/A"

                    m1, m2 = st.columns(2)
                    m1.metric(label="Total Predicted Crimes", value=int(month_total))
                    m2.metric(label="Most Frequent Crime", value=top_crime)

                    df_display = df_pred[["Crime Description", "Crime Count"]].rename(
                        columns={
                            "Crime Description": "Crime Type",
                            "Crime Count": "Count",
                        }
                    )
                    df_display = df_display[df_display["Count"] > 0].sort_values(
                        by="Count", ascending=False
                    )
                    st.dataframe(
                        df_display, width=800, hide_index=True
                    )  # width="stretch"

                    st.divider()
                    st.subheader(f"Long-Term Forecast ({selected_state})")
                    with st.spinner("Analyzing future trends..."):
                        q_val, h_val, y_val = get_long_term_forecast_from_cache(
                            selected_state,
                            selected_year,
                            selected_month,
                            selected_crime,
                        )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Next 3 Months", f"{q_val}")
                    c2.metric("Next 6 Months", f"{h_val}")
                    c3.metric("Next 1 Year", f"{y_val}")

                else:
                    st.error("No data found for this selection.")

    # ===== 2: HEATMAP ANALYSIS =====
    with tab2:
        st.subheader("State-wise Crime Intensity Heatmap")

        h_col1, h_col2 = st.columns([1, 3])

        with h_col1:
            h_year = st.number_input("Heatmap Year", 2024, 2030, 2025, key="h_year")
            h_month = st.selectbox("Heatmap Month", range(1, 13), key="h_month")
            h_crime = st.selectbox("Crime Type to Map", all_crime_types, key="h_crime")

            if selected_state:
                st.info(f"Targeting: **{selected_state}** (from Prediction Dashboard)")

            auto = st.session_state.pop("auto_update_map", False)
            gen_map = st.button("Update Heatmap") or auto

        with h_col2:
            if gen_map:
                ensure_forecast_exists(h_year + 1)
                cache = st.session_state.forecast_cache

                # Prepare Base Data
                mask = (cache["Year"] == h_year) & (cache["Month"] == h_month)
                if h_crime != "All Crimes":
                    mask &= cache["Crime Description"] == h_crime

                map_data = (
                    cache[mask].groupby("State", as_index=False)["Crime Count"].sum()
                )
                print(map_data)
                current_date = pd.to_datetime(f"{h_year}-{h_month}-01")
                date_3m = current_date + relativedelta(months=3)
                date_6m = current_date + relativedelta(months=6)
                date_1y = current_date + relativedelta(months=12)

                def get_future_sums(end_date, label):
                    temp_cache = cache.copy()
                    temp_cache["Date"] = pd.to_datetime(
                        temp_cache[["Year", "Month"]].assign(DAY=1)
                    )
                    f_mask = (temp_cache["Date"] > current_date) & (
                        temp_cache["Date"] <= end_date
                    )
                    if h_crime != "All Crimes":
                        f_mask &= temp_cache["Crime Description"] == h_crime
                    return (
                        temp_cache[f_mask]
                        .groupby("State", as_index=False)["Crime Count"]
                        .sum()
                        .rename(columns={"Crime Count": label})
                    )

                merged_df = all_states_df.merge(
                    map_data, on="State", how="left"
                ).fillna(0)
                # print(all_states_df)
                # print(merged_df)
                merged_df = merged_df.merge(
                    get_future_sums(date_3m, "Next 3 Months"), on="State", how="left"
                ).fillna(0)
                merged_df = merged_df.merge(
                    get_future_sums(date_6m, "Next 6 Months"), on="State", how="left"
                ).fillna(0)
                merged_df = merged_df.merge(
                    get_future_sums(date_1y, "Next 1 Year"), on="State", how="left"
                ).fillna(0)
                print(merged_df)
                fig = px.choropleth(
                    merged_df,
                    geojson=india_states,
                    locations="State",
                    featureidkey="properties.st_nm",
                    color="Crime Count",
                    color_continuous_scale="RdYlGn_r",
                    title=f"Crime Intensity: {h_crime} ({h_month}/{h_year})",
                    hover_data=["State", "Crime Count", "Next 3 Months"],
                )

                fig.update_traces(marker_opacity=0.5, marker_line_width=0.5)

                if selected_state and selected_state in merged_df["State"].values:
                    # Get Data
                    state_row = merged_df[merged_df["State"] == selected_state]
                    raw_val = state_row["Crime Count"]
                    crime_count = (
                        float(raw_val.iloc[0])
                        if isinstance(raw_val, pd.Series)
                        else float(raw_val)
                    )

                    # Get Location
                    center_lon, center_lat = get_state_centroid_coords(
                        selected_state, india_states
                    )

                    if center_lon and center_lat:

                        box_x_offset = 70 if center_lon < 78 else -70
                        box_y_offset = 0
                        box_x_offset = max(min(box_x_offset, 80), -80)
                        box_y_offset = max(min(box_y_offset, 80), -80)

                        if center_lat > 32:
                            box_y_offset = 50  #
                        if center_lat < 10:
                            box_y_offset = -50
                        fig.add_trace(
                            go.Choropleth(
                                geojson=india_states,
                                locations=[
                                    f["properties"]["st_nm"]
                                    for f in india_states["features"]
                                ],
                                z=[0] * len(india_states["features"]),
                                featureidkey="properties.st_nm",
                                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                showscale=False,
                                marker_line_color="black",
                                marker_line_width=1,
                                hoverinfo="skip",
                            )
                        )

                        fig.add_trace(
                            go.Scattergeo(
                                lon=[center_lon],
                                lat=[center_lat],
                                text=[
                                    f"<b>{selected_state.upper()}</b><br>Count: {int(crime_count)}"
                                ],
                                mode="markers+text",
                                textposition="top center",
                                textfont=dict(size=14, color="black"),
                                marker=dict(
                                    size=8,
                                    color="black",
                                    line=dict(width=2, color="white"),
                                ),
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                        fig.add_annotation(
                            text='<b><span style="color:red;">Red = High Crime</span><br><span style="color:green;">Green = Low Crime</span></b>',
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.8,
                            y=0.25,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(size=12),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.8,
                        )

                        # Highlight Trace
                        fig.add_trace(
                            go.Choropleth(
                                geojson=india_states,
                                locations=[selected_state],
                                z=[crime_count],
                                featureidkey="properties.st_nm",
                                colorscale="RdYlGn_r",
                                showscale=False,
                                marker_opacity=0.3,
                                marker_line_color="black",
                                marker_line_width=1.5,
                                hoverinfo="skip",
                            )
                        )

                fig.update_geos(fitbounds="locations", visible=False, bgcolor="#AEDBFE")
                fig.update_layout(height=650, margin={"r": 0, "t": 40, "l": 0, "b": 0})

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("View Raw Data (Crime Count & Future Prediction)"):

                    crime_cols = [
                        "Crime Count",
                        "Next 3 Months",
                        "Next 6 Months",
                        "Next 1 Year",
                    ]

                    merged_df = merged_df[(merged_df[crime_cols] != 0).any(axis=1)]
                    sorted_df = merged_df.sort_values("Crime Count", ascending=False)

                    marked_df = mark_danger_index(sorted_df, "Crime Count")

                    styled_df = highlight_max_row(marked_df, "Crime Count")

                    st.dataframe(styled_df)
                    st.markdown(
                        "> ℹ️ **Note:** Rows where all crime-related values are `0` have been excluded. "
                        "This reflects a **dataset limitation** where no incidents or forecasts are available for those states."
                    )


def highlight_max_row(df, column):
    max_idx = df[column].idxmax()

    return df.style.format(
        {
            "Crime Count": "{:.0f}",
            "Next 3 Months": "{:.0f}",
            "Next 6 Months": "{:.0f}",
            "Next 1 Year": "{:.0f}",
        }
    ).apply(
        lambda row: [
            (
                "background-color: #fde2e2; color: #7f1d1d; font-weight: 700;"
                if row.name == max_idx
                else ""
            )
            for _ in row
        ],
        axis=1,
    )


def mark_danger_index(df, column, symbol="🚨"):
    df = df.copy()
    max_idx = df[column].idxmax()

    new_index = df.index.astype(str).tolist()
    new_index[df.index.get_loc(max_idx)] = (
        f"{symbol} {new_index[df.index.get_loc(max_idx)]}"
    )
    df.index = new_index

    return df


def get_state_centroid_coords(state_name, geojson_data):
    target_feature = None
    for feature in geojson_data["features"]:
        if feature["properties"].get("st_nm") == state_name:
            target_feature = feature
            break
    if not target_feature:
        return None, None

    geometry = target_feature["geometry"]
    coords_list = []
    if geometry["type"] == "Polygon":
        coords_list = geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            coords_list.extend(polygon[0])

    if not coords_list:
        return None, None
    center_lon = sum(c[0] for c in coords_list) / len(coords_list)
    center_lat = sum(c[1] for c in coords_list) / len(coords_list)
    return center_lon, center_lat


def main():

    pg = st.navigation(
        [
            st.Page(visualization, title="Data Visualization"),
            st.Page(show_crime_prediction, title="Crime Prediction Model"),
        ]
    )

    pg.run()


if __name__ == "__main__":
    main()

import pandas as pd
import plotly.express as px
import streamlit as st

import api_client as api


def render_overview(filters: dict):
    st.title("FMCG Sales Overview")
    st.caption(
        "Monitor sales performance, revenue, products, and overall business trends."
    )

    summary = api.get_summary(filters)
    if summary is None:
        return

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Units Sold",
        f"{summary['total_units_sold']:,.0f}"
    )

    c2.metric(
        "Total Revenue",
        f"${summary['total_revenue']:,.0f}"
    )

    c3.metric(
        "Avg Price / Unit",
        f"${summary['avg_price_unit']:.2f}"
    )

    c4.metric(
        "Avg Daily Units",
        f"{summary['avg_daily_units']:.1f}"
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Promotion Share",
        f"{summary['promo_share_pct']:.1f}%"
    )

    c6.metric(
        "Stockout-Risk Days",
        f"{summary['stockout_risk_days_pct']:.2f}%",
        help="Share of rows where units sold exceeded that day's stock available."
    )

    c7.metric(
        "Rows in Selection",
        f"{summary['n_rows']:,}"
    )

    st.caption(
        f"Data range in selection: "
        f"{summary['date_range'][0]} → {summary['date_range'][1]}"
    )

    st.divider()

    # Sales trend
    st.subheader("Sales Trend")
    st.caption("Daily units sold for the selected filters.")

    trend = api.get_trend(filters, "day")

    if not trend.empty:
        fig = px.line(
            trend,
            x="period",
            y="units_sold",
            labels={
                "period": "Date",
                "units_sold": "Units Sold"
            }
        )

        fig.update_layout(
            height=350,
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
    else:
        st.info("No data for the current filter selection.")

    st.divider()

    # Category and brand charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top Categories")
        st.caption("Categories ranked by units sold.")

        cat = api.get_compare(
            filters,
            "category",
            top_n=10
        )

        if not cat.empty:
            fig = px.bar(
                cat,
                x="group",
                y="units_sold",
                labels={
                    "group": "Category",
                    "units_sold": "Units Sold"
                }
            )

            fig.update_layout(
                height=320,
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col_b:
        st.subheader("Top Brands")
        st.caption("Brands ranked by units sold.")

        br = api.get_compare(
            filters,
            "brand",
            top_n=10
        )

        if not br.empty:
            fig = px.bar(
                br,
                x="group",
                y="units_sold",
                labels={
                    "group": "Brand",
                    "units_sold": "Units Sold"
                }
            )

            fig.update_layout(
                height=320,
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


def render_trends(filters: dict):
    st.title("Sales Trends")
    st.caption(
        "Track sales volume, revenue, and promotional performance over time."
    )

    granularity = st.radio(
        "Granularity",
        ["day", "week", "month"],
        index=1,
        horizontal=True
    )

    trend = api.get_trend(
        filters,
        granularity
    )

    if trend.empty:
        st.info("No data for the current filter selection.")
        return

    st.subheader("Units Sold Over Time")

    fig = px.line(
        trend,
        x="period",
        y="units_sold",
        labels={
            "period": "Period",
            "units_sold": "Units Sold"
        }
    )

    fig.update_layout(
        height=400,
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Revenue Over Time")

    fig2 = px.area(
        trend,
        x="period",
        y="revenue",
        labels={
            "period": "Period",
            "revenue": "Revenue ($)"
        }
    )

    fig2.update_layout(
        height=350,
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Promotion vs Non-Promotion Sales")

    melted = trend.melt(
        id_vars="period",
        value_vars=[
            "promo_units_sold",
            "non_promo_units_sold"
        ],
        var_name="type",
        value_name="units"
    )

    melted["type"] = melted["type"].map({
        "promo_units_sold": "Promo",
        "non_promo_units_sold": "Non-promo"
    })

    fig3 = px.bar(
        melted,
        x="period",
        y="units",
        color="type",
        barmode="stack",
        labels={
            "period": "Period",
            "units": "Units Sold",
            "type": "Day Type"
        }
    )

    fig3.update_layout(
        height=350,
        margin=dict(t=20, b=20)
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


def render_compare(filters: dict):
    st.title("Sales Comparison")
    st.caption(
        "Compare products, brands, categories, regions, and sales channels."
    )

    dimension = st.selectbox(
        "Compare by",
        [
            "category",
            "brand",
            "sku",
            "region",
            "channel",
            "segment",
            "pack_type"
        ],
        index=0
    )

    top_n = st.slider(
        "Show top N",
        min_value=5,
        max_value=30,
        value=15
    )

    df = api.get_compare(
        filters,
        dimension,
        top_n
    )

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    tab1, tab2 = st.tabs([
        "Units Sold",
        "Revenue"
    ])

    with tab1:
        fig = px.bar(
            df.sort_values("units_sold"),
            x="units_sold",
            y="group",
            orientation="h",
            labels={
                "units_sold": "Units Sold",
                "group": dimension.title()
            }
        )

        fig.update_layout(
            height=max(320, 28 * len(df)),
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab2:
        fig2 = px.bar(
            df.sort_values("revenue"),
            x="revenue",
            y="group",
            orientation="h",
            labels={
                "revenue": "Revenue ($)",
                "group": dimension.title()
            }
        )

        fig2.update_layout(
            height=max(320, 28 * len(df)),
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("Details")

    st.dataframe(
        df.rename(
            columns={
                "group": dimension.title(),
                "units_sold": "Units Sold",
                "revenue": "Revenue",
                "avg_price_unit": "Avg Price/Unit",
                "n_rows": "Rows"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


def render_inventory_promotions(filters: dict):
    st.title("Inventory & Promotions")
    st.caption(
        "Analyze inventory levels, stockout risk, and promotional impact."
    )

    dimension = st.selectbox(
        "Break down by",
        [
            "category",
            "brand",
            "sku",
            "region",
            "channel"
        ],
        index=0
    )

    tab1, tab2 = st.tabs([
        "Inventory",
        "Promotions"
    ])

    with tab1:
        inv = api.get_inventory(
            filters,
            dimension
        )

        if inv.empty:
            st.info("No data for the current filter selection.")
        else:
            c1, c2 = st.columns(2)

            with c1:
                fig = px.bar(
                    inv,
                    x="group",
                    y=[
                        "avg_stock_available",
                        "avg_units_sold"
                    ],
                    barmode="group",
                    labels={
                        "group": dimension.title(),
                        "value": "Average per day",
                        "variable": "Metric"
                    }
                )

                fig.update_layout(
                    height=360,
                    margin=dict(t=20, b=20)
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with c2:
                fig2 = px.bar(
                    inv.sort_values(
                        "stockout_risk_pct",
                        ascending=False
                    ),
                    x="group",
                    y="stockout_risk_pct",
                    labels={
                        "group": dimension.title(),
                        "stockout_risk_pct": "Stockout-Risk Days (%)"
                    }
                )

                fig2.update_layout(
                    height=360,
                    margin=dict(t=20, b=20)
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            st.caption(
                "Stockout-risk days = share of rows where units sold "
                "exceeded that day's stock available."
            )

            st.dataframe(
                inv.rename(
                    columns={
                        "group": dimension.title(),
                        "avg_stock_available": "Avg Stock",
                        "avg_units_sold": "Avg Units Sold",
                        "avg_delivered_qty": "Avg Delivered Qty",
                        "avg_delivery_days": "Avg Delivery Days",
                        "stockout_risk_pct": "Stockout-Risk %"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

    with tab2:
        promo = api.get_promotions(
            filters,
            dimension
        )

        if promo.empty:
            st.info("No data for the current filter selection.")
        else:
            fig = px.bar(
                promo.sort_values("uplift_pct"),
                x="uplift_pct",
                y="group",
                orientation="h",
                labels={
                    "uplift_pct": "Promo Uplift (%)",
                    "group": dimension.title()
                }
            )

            fig.update_layout(
                height=max(320, 28 * len(promo)),
                margin=dict(t=20, b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Uplift = percentage difference in average daily units "
                "sold on promotion days vs non-promotion days."
            )

            st.dataframe(
                promo.rename(
                    columns={
                        "group": dimension.title(),
                        "promo_avg_units": "Avg Units (Promo)",
                        "non_promo_avg_units": "Avg Units (Non-promo)",
                        "uplift_pct": "Uplift %"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


def render_explore(filters: dict):
    st.title("Explore Data")
    st.caption(
        "Browse the filtered sales records used for analysis."
    )

    limit = st.slider(
        "Rows to show",
        min_value=50,
        max_value=2000,
        value=300,
        step=50
    )

    df = api.get_table(
        filters,
        limit=limit,
        offset=0
    )

    if df.empty:
        st.info("No data for the current filter selection.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="fmcg_sales_filtered.csv",
        mime="text/csv"
    )


def render_forecast():
    st.title("Sales Forecast")
    st.caption(
        "Generate a sales forecast for a selected product, region, "
        "and channel. Forecasts are model-generated estimates and "
        "should not be treated as actual sales."
    )

    series_df = api.get_forecast_series()

    if series_df.empty:
        st.error("No series available to forecast.")
        return

    col1, col2, col3 = st.columns(3)

    sku = col1.selectbox(
        "SKU",
        sorted(series_df["sku"].unique())
    )

    regions_for_sku = sorted(
        series_df.loc[
            series_df["sku"] == sku,
            "region"
        ].unique()
    )

    region = col2.selectbox(
        "Region",
        regions_for_sku
    )

    channels_for_combo = sorted(
        series_df.loc[
            (series_df["sku"] == sku)
            & (series_df["region"] == region),
            "channel"
        ].unique()
    )

    channel = col3.selectbox(
        "Channel",
        channels_for_combo
    )

    last_known = api.get_last_known(
        sku,
        region,
        channel
    )

    if last_known:
        st.info(
            f"Latest available actual data for this series is "
            f"from **{last_known['last_date']}**. "
            f"The operational values below have been prefilled "
            f"and can be adjusted for a forecast scenario."
        )

    st.subheader("Forecast Inputs")

    c1, c2 = st.columns(2)

    default_date = (
        pd.Timestamp(last_known["last_date"]).date()
        + pd.Timedelta(days=1)
        if last_known
        else None
    )

    with c1:
        target_date = st.date_input(
            "Forecast Date",
            value=default_date
        )

        price_unit = st.number_input(
            "Price per Unit ($)",
            min_value=0.01,
            value=(
                float(last_known["price_unit"])
                if last_known
                else 5.0
            ),
            step=0.1
        )

        promotion_flag = st.checkbox(
            "Promotion Running",
            value=(
                bool(last_known["promotion_flag"])
                if last_known
                else False
            )
        )

    with c2:
        delivery_days = st.number_input(
            "Delivery Days",
            min_value=0.0,
            value=(
                float(last_known["delivery_days"])
                if last_known
                else 3.0
            ),
            step=1.0
        )

        stock_available = st.number_input(
            "Stock Available (Units)",
            min_value=0.0,
            value=(
                float(last_known["stock_available"])
                if last_known
                else 150.0
            ),
            step=1.0
        )

        delivered_qty = st.number_input(
            "Delivered Quantity (Units)",
            min_value=0.0,
            value=(
                float(last_known["delivered_qty"])
                if last_known
                else 170.0
            ),
            step=1.0
        )

    if st.button(
        "Generate Forecast",
        type="primary"
    ):
        payload = {
            "target_date": str(target_date),
            "sku": sku,
            "region": region,
            "channel": channel,
            "price_unit": price_unit,
            "promotion_flag": promotion_flag,
            "delivery_days": delivery_days,
            "stock_available": stock_available,
            "delivered_qty": delivered_qty
        }

        result = api.predict(payload)

        if result:
            st.success(
                f"Forecast generated for **{sku}** in "
                f"**{region}** via **{channel}** on "
                f"**{result['date']}**."
            )

            st.metric(
                "Predicted Units Sold",
                f"{result['predicted_units_sold']:.1f}"
            )

            st.caption(
                "This is a model-generated forecast, "
                "not an actual recorded sale."
            )


def render_ai_analyst():
    """
    AI FMCG Analyst chat page.

    Uses the existing FastAPI /chat endpoint, which routes
    questions through the Groq + LangGraph agent and the
    existing sales, inventory, promotion, and forecasting tools.
    """

    st.title("🤖 AI FMCG Analyst")
    st.caption(
        "Ask questions in plain English about sales, inventory, "
        "promotions, and future forecasts."
    )

    if "ai_analyst_history" not in st.session_state:
        st.session_state.ai_analyst_history = []

    with st.expander("Example Questions"):
        st.markdown(
            "- What were Juice sales in 2024?\n"
            "- Which category had the highest sales?\n"
            "- Compare Retail and E-commerce.\n"
            "- Which region has the highest average inventory?\n"
            "- Did promotions increase sales?\n"
            "- Forecast MI-006 sales for January 1, 2025.\n"
            "- Which SKU generated the most revenue in 2024?"
        )

    for turn in st.session_state.ai_analyst_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_question = st.chat_input(
        "Ask about sales, inventory, promotions, or forecasts..."
    )

    if user_question:
        st.session_state.ai_analyst_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your question..."):
                response = api.chat(
                    user_question,
                    st.session_state.ai_analyst_history[:-1]
                )

            if response is None:
                answer = (
                    "Sorry, I couldn't reach the analytics service. "
                    "Please try again."
                )
                st.error(answer)

            else:
                answer = response.get(
                    "answer",
                    "I couldn't generate an answer for that question."
                )

                st.markdown(answer)

        st.session_state.ai_analyst_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.session_state.ai_analyst_history:
        if st.button("Clear Conversation"):
            st.session_state.ai_analyst_history = []
            st.rerun()
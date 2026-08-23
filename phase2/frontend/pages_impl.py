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


# def render_ai_analyst():
#     """
#     AI FMCG Analyst chat page.

#     Uses the existing FastAPI /chat endpoint, which routes
#     questions through the Groq + LangGraph agent and the
#     existing sales, inventory, promotion, and forecasting tools.
#     """

#     st.title("🤖 AI FMCG Analyst")
#     st.caption(
#         "Ask questions in plain English about sales, inventory, "
#         "promotions, and future forecasts."
#     )

#     if "ai_analyst_history" not in st.session_state:
#         st.session_state.ai_analyst_history = []

#     with st.expander("Example Questions"):
#         st.markdown(
#             "- What were Juice sales in 2024?\n"
#             "- Which category had the highest sales?\n"
#             "- Compare Retail and E-commerce.\n"
#             "- Which region has the highest average inventory?\n"
#             "- Did promotions increase sales?\n"
#             "- Forecast MI-006 sales for January 1, 2025.\n"
#             "- Which SKU generated the most revenue in 2024?"
#         )

#     for turn in st.session_state.ai_analyst_history:
#         with st.chat_message(turn["role"]):
#             st.markdown(turn["content"])

#     user_question = st.chat_input(
#         "Ask about sales, inventory, promotions, or forecasts..."
#     )

#     if user_question:
#         st.session_state.ai_analyst_history.append(
#             {
#                 "role": "user",
#                 "content": user_question
#             }
#         )

#         with st.chat_message("user"):
#             st.markdown(user_question)

#         with st.chat_message("assistant"):
#             with st.spinner("Analyzing your question..."):
#                 response = api.chat(
#                     user_question,
#                     st.session_state.ai_analyst_history[:-1]
#                 )

#             if response is None:
#                 answer = (
#                     "Sorry, I couldn't reach the analytics service. "
#                     "Please try again."
#                 )
#                 st.error(answer)

#             else:
#                 answer = response.get(
#                     "answer",
#                     "I couldn't generate an answer for that question."
#                 )

#                 st.markdown(answer)

#         st.session_state.ai_analyst_history.append(
#             {
#                 "role": "assistant",
#                 "content": answer
#             }
#         )

#     if st.session_state.ai_analyst_history:
#         if st.button("Clear Conversation"):
#             st.session_state.ai_analyst_history = []
            # st.rerun()

def render_ai_analyst():
    """
    Floating AI FMCG Analyst chatbot.

    The chatbot is available on every page. It opens automatically
    on the first visit and can be minimized/reopened using the
    floating robot button.
    """

    from streamlit_float import float_init

    # Initialize floating containers
    float_init()

    # ---------------------------------------------------------
    # Session state
    # ---------------------------------------------------------
    if "ai_analyst_history" not in st.session_state:
        st.session_state.ai_analyst_history = []

    if "ai_analyst_open" not in st.session_state:
        # Open automatically on first visit
        st.session_state.ai_analyst_open = True

    # ---------------------------------------------------------
    # Floating chatbot window
    # ---------------------------------------------------------
    if st.session_state.ai_analyst_open:

        chat_container = st.container()

        with chat_container:

            # Header
            header_col1, header_col2 = st.columns([8, 1])

            with header_col1:
                st.markdown(
                    """
                    <div style="
                        font-size: 1.15rem;
                        font-weight: 700;
                        padding-top: 4px;
                    ">
                        🤖 FMCG Analyst
                    </div>

                    <div style="
                        font-size: 0.78rem;
                        color: #22c55e;
                        margin-top: -4px;
                    ">
                        ● Online
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with header_col2:
                if st.button(
                    "×",
                    key="ai_close",
                    help="Close AI Analyst",
                ):
                    st.session_state.ai_analyst_open = False
                    st.rerun()

            st.divider()

            # -------------------------------------------------
            # Welcome message
            # -------------------------------------------------
            if not st.session_state.ai_analyst_history:
                st.markdown(
                    """
                    <div style="
                        background: #f3f6fb;
                        border-radius: 14px;
                        padding: 14px;
                        margin-bottom: 10px;
                    ">
                        <div style="
                            font-size: 1rem;
                            font-weight: 600;
                            margin-bottom: 6px;
                        ">
                            Hi! 👋 I'm your FMCG Analyst
                        </div>

                        <div style="
                            font-size: 0.88rem;
                            line-height: 1.45;
                        ">
                            I can help you analyze sales, revenue,
                            inventory, promotions, and forecasts.
                        </div>

                        <div style="
                            font-size: 0.88rem;
                            font-weight: 600;
                            margin-top: 10px;
                        ">
                            What would you like to know?
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.caption("💡 Try asking:")

                example_questions = [
                    "What were the total sales in 2024?",
                    "Which category had the highest sales?",
                    "Which region has the highest average inventory?",
                    "Did promotions increase sales?",
                ]

                for i, question in enumerate(example_questions):
                    if st.button(
                        question,
                        key=f"ai_example_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.ai_pending_question = question
                        st.rerun()

            # -------------------------------------------------
            # Conversation
            # -------------------------------------------------
            history_container = st.container(height=280)

            with history_container:
                for turn in st.session_state.ai_analyst_history:
                    with st.chat_message(
                        turn["role"],
                        avatar="🤖" if turn["role"] == "assistant" else "👤",
                    ):
                        st.markdown(turn["content"])

            # -------------------------------------------------
            # Chat input
            # -------------------------------------------------
            user_question = st.chat_input(
                "Ask about sales, revenue, inventory or forecasts...",
                key="ai_chat_input",
            )

            # Example question selected
            if (
                "ai_pending_question" in st.session_state
                and st.session_state.ai_pending_question
            ):
                user_question = st.session_state.ai_pending_question
                st.session_state.ai_pending_question = None

            # -------------------------------------------------
            # Process question
            # -------------------------------------------------
            if user_question:

                st.session_state.ai_analyst_history.append(
                    {
                        "role": "user",
                        "content": user_question,
                    }
                )

                with st.spinner("Analyzing your question..."):

                    response = api.chat(
                        user_question,
                        st.session_state.ai_analyst_history[:-1],
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
                        "I couldn't generate an answer for that question.",
                    )

                st.session_state.ai_analyst_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.rerun()

            # -------------------------------------------------
            # Footer
            # -------------------------------------------------
            footer_col1, footer_col2 = st.columns([5, 2])

            with footer_col1:
                st.caption("⚡ Powered by Groq + LangGraph")

            with footer_col2:
                if st.button(
                    "Clear",
                    key="ai_clear",
                    use_container_width=True,
                ):
                    st.session_state.ai_analyst_history = []
                    st.rerun()

        # Float the entire chat window
        chat_container.float(
            """
            width: 390px;
            max-width: calc(100vw - 2rem);
            right: 1rem;
            bottom: 5.2rem;
            z-index: 9999;
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.12);
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 10px 35px rgba(0, 0, 0, 0.22);
            """
        )

    # ---------------------------------------------------------
    # Floating robot launcher
    # ---------------------------------------------------------

    launcher = st.container()

    with launcher:

        if st.button(
            "🤖",
            key="ai_launcher",
            help="Open FMCG Analyst",
        ):
            st.session_state.ai_analyst_open = True
            st.rerun()

    launcher.float(
        """
        width: 58px;
        height: 58px;
        right: 1.2rem;
        bottom: 1.8rem;
        z-index: 10000;
        border-radius: 50%;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.40);
        padding: 0;
        text-align: center;
        """
    )

    # ---------------------------------------------------------
    # Robot button styling
    # ---------------------------------------------------------
    st.markdown(
        """
        <style>

        /* Floating robot button */
        div[data-testid="stButton"] button[kind="secondary"] {
            border-radius: 50%;
        }

        /* Make the robot emoji larger */
        div[data-testid="stButton"] button {
            font-size: 1.45rem;
        }

        /* Chat window buttons */
        div[data-testid="stVerticalBlock"] button {
            border-radius: 10px;
        }

        /* Mobile adjustment */
        @media (max-width: 600px) {
            .floating {
                right: 0.6rem !important;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )     
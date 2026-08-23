import streamlit as st

import api_client as api
import pages_impl as pg

st.set_page_config(page_title="FMCG Sales Analytics", page_icon="📦", layout="wide")


def render_sidebar_filters(opts: dict) -> dict:
    st.sidebar.title("📦 FMCG Sales Analytics")
    st.sidebar.caption("Sales analytics & forecasting platform")
    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    date_range = st.sidebar.date_input(
        "Date range",
        value=(opts["min_date"], opts["max_date"]),
        min_value=opts["min_date"],
        max_value=opts["max_date"],
    )
    start_date, end_date = (date_range if len(date_range) == 2 else (opts["min_date"], opts["max_date"]))

    categories = st.sidebar.multiselect("Category", opts["categories"])
    brands = st.sidebar.multiselect("Brand", opts["brands"])
    regions = st.sidebar.multiselect("Region", opts["regions"])
    channels = st.sidebar.multiselect("Channel", opts["channels"])
    skus = st.sidebar.multiselect("SKU", opts["skus"])
    segments = st.sidebar.multiselect("Segment", opts["segments"])

    promo_choice = st.sidebar.radio("Promotion status", ["All days", "Promo days only", "Non-promo days only"], index=0)
    promotion_only = {"All days": None, "Promo days only": True, "Non-promo days only": False}[promo_choice]

    st.sidebar.divider()
    if st.sidebar.button("Reset filters"):
        st.rerun()

    return {
        "start_date": str(start_date), "end_date": str(end_date),
        "categories": categories, "brands": brands, "regions": regions,
        "channels": channels, "skus": skus, "segments": segments,
        "promotion_only": promotion_only,
    }


def main():
    opts = api.get_filter_options()
    filters = render_sidebar_filters(opts)

    # page = st.sidebar.radio(
    #     "Go to",
    #     ["Overview", "Sales Trends", "Compare", "Inventory & Promotions", "Explore Data",
    #      "Forecast", "AI FMCG Analyst"],
    #     index=0,
    # )

    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Sales Trends", "Compare", "Inventory & Promotions", "Explore Data",
         "Forecast"],
         index=0,
    )

    if page == "Overview":
        pg.render_overview(filters)
    elif page == "Sales Trends":
        pg.render_trends(filters)
    elif page == "Compare":
        pg.render_compare(filters)
    elif page == "Inventory & Promotions":
        pg.render_inventory_promotions(filters)
    elif page == "Explore Data":
        pg.render_explore(filters)
    elif page == "Forecast":
        pg.render_forecast()
    # elif page == "AI FMCG Analyst":
    #     pg.render_ai_analyst()
    pg.render_ai_analyst()


if __name__ == "__main__":
    main()

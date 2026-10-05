"""
Small Business Sales Analyzer
Libraries: Streamlit, Pandas, NumPy, Matplotlib, SciPy
Run locally:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Sales Analyzer", layout="wide")

DEFAULT_CSV = "sales.csv"
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_and_clean(source):
    """Read the CSV, fix types, drop invalid rows, and add derived columns."""
    raw = pd.read_csv(source)
    df = raw.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    
    missing_payment = int(df["payment_method"].isna().sum())
    df["payment_method"] = df["payment_method"].fillna("Unknown")

    
    df = df.dropna(subset=["date", "product", "category", "quantity", "unit_price"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]

    df["total"] = df["quantity"] * df["unit_price"]
    df["weekday"] = pd.Categorical(df["date"].dt.day_name(), categories=WEEKDAY_ORDER, ordered=True)
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    report = {
        "rows_read": len(raw),
        "rows_kept": len(df),
        "rows_dropped": len(raw) - len(df),
        "payment_filled": missing_payment,
    }
    return df.reset_index(drop=True), report



st.sidebar.title("Sales Analyzer")
try:
    data, clean_report = load_and_clean(DEFAULT_CSV)
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()


st.sidebar.header("Filters")
min_d, max_d = data["date"].min().date(), data["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
else:
    start, end = min_d, max_d

categories = st.sidebar.multiselect(
    "Category", sorted(data["category"].unique()), default=sorted(data["category"].unique())
)
payments = st.sidebar.multiselect(
    "Payment method", sorted(data["payment_method"].unique()), default=sorted(data["payment_method"].unique())
)


mask = (
    (data["date"].dt.date >= start)
    & (data["date"].dt.date <= end)
    & (data["category"].isin(categories))
    & (data["payment_method"].isin(payments))
)
df = data[mask]

st.title("Small Business Sales Analyzer")


if df.empty:
    st.warning("No records match the current filters. Adjust the sidebar filters.")
    st.stop()

totals = df["total"].to_numpy()

tab_overview, tab_data, tab_charts, tab_stats, tab_about = st.tabs(
    ["Overview", "Data", "Charts", "Statistics", "About"]
)


with tab_overview:
    best_product = df.groupby("product")["total"].sum().idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sales", f"₱{np.sum(totals):,.2f}")
    c2.metric("Average order", f"₱{np.mean(totals):,.2f}")
    c3.metric("Orders", f"{len(df)}")
    c4.metric("Top product", best_product)

    st.subheader("Order value statistics")
    stat_table = pd.DataFrame(
        {
            "Statistic": ["Mean", "Median", "Standard deviation", "Minimum", "Maximum", "Range"],
            "Value (₱)": [
                np.mean(totals),
                np.median(totals),
                np.std(totals),
                np.min(totals),
                np.max(totals),
                np.ptp(totals),
            ],
        }
    )
    st.dataframe(stat_table.style.format({"Value (₱)": "{:,.2f}"}), hide_index=True, use_container_width=True)

    st.subheader("Share of sales by category")
    cat_sales = df.groupby("category")["total"].sum().sort_values(ascending=False)
    share = cat_sales.to_numpy() / np.sum(cat_sales.to_numpy()) * 100
    summary = pd.DataFrame(
        {"Category": cat_sales.index, "Sales (₱)": cat_sales.values, "Share (%)": np.round(share, 1)}
    )
    st.dataframe(summary.style.format({"Sales (₱)": "{:,.2f}"}), hide_index=True, use_container_width=True)


with tab_data:
    st.subheader("Filtered records")
    sort_col = st.selectbox("Sort by", ["date", "total", "quantity", "product"], index=1)
    ascending = st.toggle("Ascending", value=False)
    view = df.sort_values(sort_col, ascending=ascending)
    st.dataframe(view.drop(columns=["is_weekend"]), hide_index=True, use_container_width=True)
    st.download_button("Download filtered data", view.to_csv(index=False), "filtered_sales.csv", "text/csv")

    st.subheader("Sales by product")
    by_product = (
        df.groupby("product")
        .agg(orders=("order_id", "count"), units=("quantity", "sum"), sales=("total", "sum"))
        .sort_values("sales", ascending=False)
        .reset_index()
    )
    st.dataframe(by_product.style.format({"sales": "{:,.2f}"}), hide_index=True, use_container_width=True)


with tab_charts:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**1. Sales by category**")
        fig, ax = plt.subplots(figsize=(6, 4))
        cat_sales = df.groupby("category")["total"].sum().sort_values(ascending=False)
        ax.bar(cat_sales.index, cat_sales.values, color="#4C78A8")
        ax.set_ylabel("Sales (₱)")
        ax.set_xlabel("Category")
        st.pyplot(fig)
        top_cat = cat_sales.idxmax()
        st.caption(
            f"Shows which categories earn the most. **{top_cat}** leads with "
            f"₱{cat_sales.max():,.0f}, so it is the main revenue driver."
        )

    with col_b:
        st.markdown("**2. Daily sales trend**")
        fig, ax = plt.subplots(figsize=(6, 4))
        daily = df.groupby(df["date"].dt.date)["total"].sum()
        ax.plot(daily.index, daily.values, marker="o", markersize=3, color="#F58518")
        ax.set_ylabel("Sales (₱)")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        st.pyplot(fig)
        st.caption(
            f"this Shows day to day changes in revenue. The best day was **{daily.idxmax()}** "
            f"(₱{daily.max():,.0f})"
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**3. Order value distribution**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(totals, bins=10, color="#54A24B", edgecolor="white")
        ax.axvline(np.mean(totals), color="red", linestyle="--", label=f"Mean ₱{np.mean(totals):,.0f}")
        ax.set_xlabel("Order value (₱)")
        ax.set_ylabel("Number of orders")
        ax.legend()
        st.pyplot(fig)
        st.caption(
            "Shows how order sizes ate spread. Most orders fall in tje lower range, "
            "while a few large orders pull the mean upward."
        )

    with col_d:
        st.markdown("**4. Payment method share**")
        fig, ax = plt.subplots(figsize=(6, 4))
        pay_counts = df["payment_method"].value_counts()
        ax.pie(pay_counts.values, labels=pay_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
        st.caption(
            f"Shows how customers pay. **{pay_counts.idxmax()}** is the most commmon method "
            f"({pay_counts.max() / pay_counts.sum() * 100:.0f}% of orders)."
        )

    st.markdown("**5. Average sales by weekday**")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    per_day = df.groupby([df["date"].dt.date, "weekday"], observed=True)["total"].sum().reset_index()
    weekday_avg = per_day.groupby("weekday", observed=True)["total"].mean()
    ax.bar(weekday_avg.index.astype(str), weekday_avg.values, color="#B279A2")
    ax.set_ylabel("Avg daily sales (₱)")
    st.pyplot(fig)
    st.caption("Compares typical daily revenue for each day of the week.")

with tab_stats:
    st.subheader("Statistical analysis")

    st.markdown("**A. Correlation: quantity vs. order total**")
    if len(df) >= 3 and df["quantity"].nunique() > 1:
        r, p = stats.pearsonr(df["quantity"], df["total"])
        st.write(f"Pearson r = **{r:.3f}**, p-value = **{p:.4f}**")
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        significance = "statistically significant" if p < 0.05 else "not statistically significant"
        st.info(
            f"There is a {strength} {direction} relationship between quantity and order total, "
            f"and it is {significance} at the 5% level."
        )
    else:
        st.warning("Not enough varied data to compute a correlation with the current filters.")

    st.markdown("**B. Weekday vs. weekend daily sales (independent t-test)**")
    daily_all = df.groupby(df["date"].dt.date).agg(total=("total", "sum"), weekend=("is_weekend", "first"))
    weekend_sales = daily_all.loc[daily_all["weekend"], "total"]
    weekday_sales = daily_all.loc[~daily_all["weekend"], "total"]

    if len(weekend_sales) >= 2 and len(weekday_sales) >= 2:
        t_stat, p_val = stats.ttest_ind(weekend_sales, weekday_sales, equal_var=False)
        m1, m2 = st.columns(2)
        m1.metric("Avg weekend day", f"₱{weekend_sales.mean():,.2f}")
        m2.metric("Avg weekday", f"₱{weekday_sales.mean():,.2f}")
        st.write(f"t = **{t_stat:.3f}**, p-value = **{p_val:.4f}**")
        if p_val < 0.05:
            higher = "weekends" if weekend_sales.mean() > weekday_sales.mean() else "weekdays"
            st.success(f"The difference is statistically significant: **{higher}** earn more per day.")
        else:
            st.info("The difference is not statistically significant, so we cannot say weekends sell differently.")
    else:
        st.warning("Need at least two weekend days and two weekdays in the filtered range.")


with tab_about:
    st.subheader("Dataset documentation")
    st.markdown(
        f"""
- **Source:** we manually inputed datas for a small coffee shop.
- **Records:** {clean_report['rows_read']} read, {clean_report['rows_kept']} kept after cleaning
  ({clean_report['rows_dropped']} dropped, {clean_report['payment_filled']} missing payment methods labeled "Unknown").
- **Data type:** Structured transactional data (categorical, numeric, and date fields).

| Column | Meaning |
|---|---|
| order_id | Unique order number |
| date | Date of the sale |
| product | Item sold |
| category | Coffee, Tea, Pastry, or Food |
| quantity | Units sold in the order |
| unit_price | Price per unit  |
| payment_method | Cash, GCash, or Card |
| total | quantity × unit_price |
"""
    )
    st.subheader("Libraries used")
    st.markdown(
        """
- **Pandas:** load CSV and clean and filter data.
- **NumPy:** we used this for computing mean, median, standard deviation, min, max, range, and percentage share.
- **Matplotlib:** we used this for visualizing data such as bar, line, histogram, and pie charts.
- **SciPy:** Pearson correlation and Welch's t-test.
- **Streamlit:** we used this for creating the interactive web interface.
"""
    )

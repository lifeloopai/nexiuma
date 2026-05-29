"""Nexiuma Streamlit dashboard for interactive backtesting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.engine import BacktestEngine
from config.settings import load_settings
from research.walkforward import WalkForwardEngine
from research.walkforward_report import WalkForwardReportGenerator
from research.walkforward_universe import WalkForwardUniverseEngine
from research.walkforward_universe_report import WalkForwardUniverseReportGenerator
from reports.generator import ReportGenerator
from strategies.registry import list_strategies


st.set_page_config(
    page_title="Nexiuma",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _plot_equity_plotly(series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Equity",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400,
    )
    return fig


def _plot_drawdown_plotly(equity: pd.Series) -> go.Figure:
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#dc2626"),
        )
    )
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=300,
    )
    return fig


def _plot_price_plotly(result) -> go.Figure:
    close = result.ohlcv["close"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=close.index, y=close.values, mode="lines", name="Close")
    )
    if result.buy_signals:
        buy_idx = [close.index.get_indexer([t], method="nearest")[0] for t in result.buy_signals]
        fig.add_trace(
            go.Scatter(
                x=[close.index[i] for i in buy_idx if i >= 0],
                y=[close.iloc[i] for i in buy_idx if i >= 0],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#16a34a"),
            )
        )
    if result.sell_signals:
        sell_idx = [close.index.get_indexer([t], method="nearest")[0] for t in result.sell_signals]
        fig.add_trace(
            go.Scatter(
                x=[close.index[i] for i in sell_idx if i >= 0],
                y=[close.iloc[i] for i in sell_idx if i >= 0],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#dc2626"),
            )
        )
    fig.update_layout(title=f"{result.ticker} Price", template="plotly_white", height=400)
    return fig


def main() -> None:
    st.title("Nexiuma")
    st.caption("Quantitative Research & Systematic Trading Platform")

    strategies = list_strategies()
    strategy_names = [s["name"] for s in strategies]

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Ticker", value="AAPL").upper()
        strategy = st.selectbox("Strategy", strategy_names, index=0)
        fast_period = st.number_input(
            "Fast SMA Period",
            min_value=2,
            max_value=500,
            value=20,
            step=1,
            disabled=strategy != "moving_average",
            help="Fast moving-average lookback (moving_average only)",
        )
        slow_period = st.number_input(
            "Slow SMA Period",
            min_value=3,
            max_value=500,
            value=50,
            step=1,
            disabled=strategy != "moving_average",
            help="Slow moving-average lookback (moving_average only)",
        )
        start = st.date_input("Start Date", value=pd.Timestamp("2020-01-01"))
        end = st.date_input("End Date", value=pd.Timestamp("2024-12-31"))
        capital = st.number_input("Initial Capital", value=100_000.0, min_value=1000.0)
        stop_loss = st.slider("Stop Loss %", 0.01, 0.20, 0.05, 0.01)
        take_profit = st.slider("Take Profit %", 0.05, 0.50, 0.15, 0.01)
        position_size = st.slider("Position Size %", 0.1, 1.0, 0.95, 0.05)
        refresh = st.checkbox("Refresh data cache", value=False)
        run_btn = st.button("Run Backtest", type="primary", use_container_width=True)

    tab_run, tab_wf, tab_wf_uni, tab_reports = st.tabs(
        ["Backtest", "Walk-Forward Analysis", "Walk-Forward Universe", "Historical Reports"]
    )

    with tab_run:
        if run_btn:
            if strategy == "moving_average" and fast_period >= slow_period:
                st.error("Fast period must be less than slow period (both > 1).")
            else:
                with st.spinner("Running backtest..."):
                    overrides: dict = {
                        "ticker": ticker,
                        "strategy": strategy,
                        "start_date": start,
                        "end_date": end,
                        "initial_capital": capital,
                        "stop_loss_pct": stop_loss,
                        "take_profit_pct": take_profit,
                        "position_size_pct": position_size,
                        "auto_refresh": refresh,
                    }
                    if strategy == "moving_average":
                        overrides["fast_period"] = int(fast_period)
                        overrides["slow_period"] = int(slow_period)
                    settings = load_settings(overrides=overrides)
                    runner = BacktestEngine(settings)
                    result = runner.run(strategy_name=strategy, ticker=ticker)
                    report_path = ReportGenerator(settings).generate(result)
                    st.session_state["last_result"] = result
                    st.session_state["last_report"] = str(report_path)
                    st.success(f"Backtest complete. Report: {report_path.name}")

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            m = result.performance.metrics
            r = result.performance.risk

            if result.strategy_name == "moving_average" and result.strategy_params:
                fp = result.strategy_params.get("fast_period")
                sp = result.strategy_params.get("slow_period")
                if fp is not None and sp is not None:
                    st.caption(f"Parameters: Fast SMA **{fp}** / Slow SMA **{sp}**")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{m.total_return:.2%}")
            col2.metric("Sharpe", f"{m.sharpe_ratio:.2f}")
            col3.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
            col4.metric("Trades", m.num_trades)

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Annualized Return", f"{m.annualized_return:.2%}")
            col6.metric("Sortino", f"{m.sortino_ratio:.2f}")
            col7.metric("Win Rate", f"{m.win_rate:.1%}")
            col8.metric("Profit Factor", f"{m.profit_factor:.2f}")

            st.subheader("Risk")
            st.json(r.to_dict())

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_plot_equity_plotly(result.equity_curve), use_container_width=True)
            with c2:
                st.plotly_chart(_plot_drawdown_plotly(result.equity_curve), use_container_width=True)
            st.plotly_chart(_plot_price_plotly(result), use_container_width=True)

            if st.session_state.get("last_report"):
                st.info(f"Full HTML report saved to: {st.session_state['last_report']}")
        else:
            st.info("Configure parameters in the sidebar and click **Run Backtest**.")

    with tab_wf:
        st.subheader("Walk-Forward Analysis")
        st.caption(
            "Optimize parameters on rolling train windows, then validate on unseen test periods."
        )
        wf_col1, wf_col2, wf_col3 = st.columns(3)
        with wf_col1:
            wf_ticker = st.text_input("WF Ticker", value="AAPL", key="wf_ticker").upper()
        with wf_col2:
            wf_train = st.number_input(
                "Train Years", min_value=1, max_value=10, value=3, key="wf_train"
            )
        with wf_col3:
            wf_test = st.number_input(
                "Test Years", min_value=1, max_value=5, value=1, key="wf_test"
            )
        wf_strategy = st.selectbox(
            "WF Strategy",
            strategy_names,
            index=0,
            key="wf_strategy",
        )
        wf_run = st.button("Run Walk-Forward", type="primary", key="wf_run_btn")

        if wf_run:
            if wf_strategy != "moving_average":
                st.error("Walk-forward currently supports moving_average only.")
            else:
                with st.spinner("Running walk-forward analysis..."):
                    wf_settings = load_settings(
                        overrides={
                            "ticker": wf_ticker,
                            "strategy": wf_strategy,
                            "start_date": start,
                            "end_date": end,
                            "initial_capital": capital,
                        }
                    )
                    wf_result = WalkForwardEngine(wf_settings).run(
                        ticker=wf_ticker,
                        strategy_name=wf_strategy,
                        train_years=int(wf_train),
                        test_years=int(wf_test),
                    )
                    wf_paths = WalkForwardReportGenerator(wf_settings).publish(wf_result)
                    st.session_state["wf_result"] = wf_result
                    st.session_state["wf_report"] = str(wf_paths["summary"])
                    st.success(f"Walk-forward complete. Report: {wf_paths['summary'].name}")

        if "wf_result" in st.session_state:
            wf_result = st.session_state["wf_result"]
            rob = wf_result.robustness
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg Test Sharpe", f"{rob.avg_test_sharpe:.2f}")
            m2.metric("Avg Test Return", f"{rob.avg_test_return:.2%}")
            m3.metric("Worst Drawdown", f"{rob.worst_test_drawdown:.2%}")
            m4.metric("Profitable Windows", f"{rob.profitable_windows_pct:.0%}")

            st.dataframe(wf_result.results_df, use_container_width=True)

            if not wf_result.combined_equity.empty:
                st.plotly_chart(
                    _plot_equity_plotly(wf_result.combined_equity),
                    use_container_width=True,
                )

            st.subheader("Parameter History")
            st.line_chart(
                wf_result.parameter_history_df.set_index("window_id")[
                    ["fast_period", "slow_period"]
                ]
            )

            if st.session_state.get("wf_report"):
                report_path = Path(st.session_state["wf_report"])
                if report_path.exists():
                    st.download_button(
                        "Download Summary HTML",
                        data=report_path.read_bytes(),
                        file_name=report_path.name,
                        mime="text/html",
                    )
                    st.info(f"Full report: {report_path}")
        else:
            st.info("Configure walk-forward settings and click **Run Walk-Forward**.")

    with tab_wf_uni:
        st.subheader("Walk-Forward Universe")
        st.caption(
            "Run walk-forward analysis across multiple assets and aggregate robustness."
        )
        uni_default = "AAPL,AMZN,MSFT,META,NVDA,GOOGL"
        wf_uni_tickers = st.text_input(
            "Universe Tickers (comma-separated)",
            value=uni_default,
            key="wf_uni_tickers",
        )
        u1, u2, u3 = st.columns(3)
        with u1:
            wf_uni_train = st.number_input(
                "Train Years", min_value=1, max_value=10, value=3, key="wf_uni_train"
            )
        with u2:
            wf_uni_test = st.number_input(
                "Test Years", min_value=1, max_value=5, value=1, key="wf_uni_test"
            )
        with u3:
            wf_uni_strategy = st.selectbox(
                "Strategy", strategy_names, index=0, key="wf_uni_strategy"
            )
        wf_uni_run = st.button(
            "Run Universe Analysis", type="primary", key="wf_uni_run_btn"
        )

        if wf_uni_run:
            if wf_uni_strategy != "moving_average":
                st.error("Walk-forward universe supports moving_average only.")
            else:
                tickers = [t.strip().upper() for t in wf_uni_tickers.split(",") if t.strip()]
                with st.spinner(f"Running walk-forward on {len(tickers)} assets..."):
                    uni_settings = load_settings(
                        overrides={
                            "strategy": wf_uni_strategy,
                            "start_date": start,
                            "end_date": end,
                            "initial_capital": capital,
                        }
                    )
                    uni_result = WalkForwardUniverseEngine(uni_settings).run(
                        strategy_name=wf_uni_strategy,
                        tickers=tickers,
                        train_years=int(wf_uni_train),
                        test_years=int(wf_uni_test),
                    )
                    uni_paths = WalkForwardUniverseReportGenerator(uni_settings).publish(
                        uni_result
                    )
                    st.session_state["wf_uni_result"] = uni_result
                    st.session_state["wf_uni_report"] = str(uni_paths["index"])
                    st.success(f"Universe analysis complete. Score: {uni_result.robustness_score.score:.0f}/100")

        if "wf_uni_result" in st.session_state:
            uni = st.session_state["wf_uni_result"]
            rs = uni.robustness_score
            sm = uni.summary_metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Robustness Score", f"{rs.score:.0f}/100")
            c2.metric("Grade", rs.grade)
            c3.metric("Avg Test Sharpe", f"{sm.avg_test_sharpe:.2f}")
            c4.metric("Positive OOS Sharpe", f"{sm.pct_positive_test_sharpe:.0%}")

            st.write(uni.executive_summary)
            st.dataframe(uni.asset_results_df, use_container_width=True)

            st.subheader("Parameter Frequency")
            st.bar_chart(uni.parameter_frequency_df.set_index("params")["frequency_pct"])

            if st.session_state.get("wf_uni_report"):
                rp = Path(st.session_state["wf_uni_report"])
                if rp.exists():
                    st.download_button(
                        "Download Universe Report",
                        data=rp.read_bytes(),
                        file_name=rp.name,
                        mime="text/html",
                    )
        else:
            st.info("Configure universe settings and click **Run Universe Analysis**.")

    with tab_reports:
        settings = load_settings()
        reports = ReportGenerator(settings).list_reports()
        if not reports:
            st.write("No historical reports yet. Run a backtest first.")
        else:
            for rep in reports:
                perf = rep.get("performance", {})
                with st.expander(
                    f"{rep.get('strategy', '?')} / {rep.get('ticker', '?')} — {rep.get('run_id', '')}"
                ):
                    st.write(f"Period: {rep.get('start_date')} → {rep.get('end_date')}")
                    if perf:
                        st.metric("Total Return", f"{perf.get('total_return', 0):.2%}")
                    path = rep.get("path")
                    if path and Path(path).exists():
                        st.markdown(f"[Open Report](file://{Path(path).resolve()})")


if __name__ == "__main__":
    main()

import streamlit as st
import os
import json
import uuid
import time
import subprocess
import sys
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame, APIError
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ta

# ── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Quant Trading Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS for a clean, professional look ────────────────────────
st.markdown("""
<style>
    /* Tighten default padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    /* Muted subheader colour */
    h2 { color: #e0e0e0; }
    /* Metric label sizing */
    [data-testid="metric-container"] label { font-size: 0.78rem; }
    /* Thinner section dividers */
    hr { margin: 0.6rem 0; border-color: #333; }
</style>
""", unsafe_allow_html=True)

# ── API connection ────────────────────────────────────────────────────────────
load_dotenv()
ALPACA_API_KEY  = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET   = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL')

if not (ALPACA_API_KEY and ALPACA_SECRET and ALPACA_BASE_URL):
    st.error("Alpaca API keys not found. Check your .env file.")
    st.stop()

@st.cache_resource
def get_api():
    return REST(ALPACA_API_KEY, ALPACA_SECRET, ALPACA_BASE_URL)

api = get_api()

# ── Helper: safely call API, return None on failure ──────────────────────────
def safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"API error: {e}")
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Quant Trading Dashboard")
    st.caption("Paper trading — Alpaca Markets")
    st.divider()

    st.subheader("Emergency Controls")
    st.warning("Use these to immediately exit all market activity.")

    if st.button("Cancel All Active Orders", width='stretch'):
        try:
            api.cancel_all_orders()
            st.success("All open orders cancelled.")
        except Exception as e:
            st.error(f"Failed: {e}")

    if st.button("CLOSE ALL POSITIONS", width='stretch', type="primary"):
        try:
            closed = api.close_all_positions()
            st.success(f"Closed {len(closed)} position(s). Back to cash.")
        except Exception as e:
            st.error(f"Failed: {e}")

    st.divider()

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# ── Fetch account state ───────────────────────────────────────────────────────
account   = safe_call(api.get_account)
positions = safe_call(api.list_positions) or []
orders    = safe_call(api.list_orders, status='all', limit=25) or []

# ── Title row ─────────────────────────────────────────────────────────────────
st.title("Quant Trading Dashboard")
if account:
    status_colour = "green" if account.status == "ACTIVE" else "red"
    st.markdown(
        f"Account status: :{status_colour}[{account.status}] &nbsp;|&nbsp; "
        f"Paper trading account",
        unsafe_allow_html=False,
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_positions, tab_orders, tab_risk, tab_chart, tab_manual, tab_logs, tab_wf = st.tabs([
    "Overview",
    "Positions",
    "Orders",
    "Risk Monitor",
    "Price Chart",
    "Manual Order",
    "Bot Logs",
    "Walk-Forward",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if account:
        equity        = float(account.equity)
        buying_power  = float(account.buying_power)
        cash          = float(account.cash)
        pv            = float(account.portfolio_value)
        dt_count      = int(account.daytrade_count)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Equity",        f"${equity:,.2f}")
        c2.metric("Portfolio Value", f"${pv:,.2f}")
        c3.metric("Cash",          f"${cash:,.2f}")
        c4.metric("Buying Power",  f"${buying_power:,.2f}")
        c5.metric("Day Trades (PDT)", str(dt_count))

        st.divider()

    # Portfolio equity history chart + drawdown
    st.subheader("Portfolio Equity History")
    days_slider = st.slider("History window (days)", 7, 365, 90, key="overview_days")

    try:
        raw_hist = api.get_portfolio_history(period=f"{days_slider}D", timeframe="1D")
        hist_df = pd.DataFrame({"timestamp": raw_hist.timestamp, "equity": raw_hist.equity})
        hist_df['dt'] = pd.to_datetime(hist_df['timestamp'], unit='s', utc=True).dt.tz_convert('America/New_York')
        hist_df = hist_df.set_index('dt').sort_index()

        # Append live equity as "now" point
        if account and not hist_df.empty:
            now_ts = pd.Timestamp.now(tz='America/New_York')
            if hist_df.index[-1].date() < now_ts.date():
                hist_df = pd.concat([hist_df, pd.DataFrame({"equity": [equity]}, index=[now_ts])])
            else:
                hist_df.iloc[-1, hist_df.columns.get_loc('equity')] = equity

        hist_df['peak']     = hist_df['equity'].cummax()
        hist_df['drawdown'] = (hist_df['equity'] - hist_df['peak']) / hist_df['peak'] * 100
        max_dd = hist_df['drawdown'].min()

        # Metrics row
        first_eq = hist_df['equity'].iloc[0]
        total_ret = (equity - first_eq) / first_eq * 100 if first_eq else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return (period)", f"{total_ret:+.2f}%")
        m2.metric("Peak Equity",           f"${hist_df['peak'].max():,.2f}")
        m3.metric("Max Drawdown",          f"{max_dd:.2f}%", delta=f"{max_dd:.2f}%", delta_color="inverse")

        # Equity + drawdown chart (two-panel)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['equity'],
            mode='lines', name='Equity',
            line=dict(color='#4fc3f7', width=2),
            fill='tozeroy', fillcolor='rgba(79,195,247,0.08)'
        ))
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['drawdown'],
            mode='lines', name='Drawdown %',
            line=dict(color='#ef5350', width=1.5),
            yaxis='y2'
        ))
        fig.update_layout(
            height=380,
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation='h', y=1.05),
            yaxis=dict(title='Equity ($)', domain=[0.35, 1.0]),
            yaxis2=dict(title='Drawdown %', domain=[0.0, 0.30], ticksuffix='%'),
            xaxis=dict(rangeslider_visible=False),
            hovermode='x unified',
        )
        st.plotly_chart(fig, width='stretch')

        # Daily P/L bar chart
        hist_df['daily_pnl_pct'] = hist_df['equity'].pct_change() * 100
        fig_pnl = go.Figure(go.Bar(
            x=hist_df.index,
            y=hist_df['daily_pnl_pct'],
            marker_color=['#26a69a' if v >= 0 else '#ef5350' for v in hist_df['daily_pnl_pct']],
            name='Daily P/L %',
        ))
        fig_pnl.update_layout(
            title="Daily P/L %",
            height=220, margin=dict(t=30, b=20, l=10, r=10),
            xaxis=dict(rangeslider_visible=False),
            yaxis=dict(ticksuffix='%'),
            hovermode='x unified',
        )
        st.plotly_chart(fig_pnl, width='stretch')

    except Exception as e:
        st.error(f"Could not load portfolio history: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — POSITIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.subheader("Open Positions")

    if not positions:
        st.info("No open positions.")
    else:
        rows = []
        for p in positions:
            qty         = int(float(p.qty))
            entry       = float(p.avg_entry_price)
            current     = float(p.current_price)
            mkt_val     = float(p.market_value)
            unreal_pl   = float(p.unrealized_pl)
            unreal_pct  = float(p.unrealized_plpc) * 100
            side        = "Long" if qty > 0 else "Short"
            rows.append({
                "Symbol":        p.symbol,
                "Side":          side,
                "Qty":           qty,
                "Entry Price":   entry,
                "Current Price": current,
                "Market Value":  mkt_val,
                "Unrealized P/L":unreal_pl,
                "P/L %":         unreal_pct,
            })

        df_pos = pd.DataFrame(rows)

        # Colour-code P/L column
        def colour_pl(val):
            colour = '#26a69a' if val >= 0 else '#ef5350'
            return f'color: {colour}'

        styled = df_pos.style\
            .format({
                "Entry Price":    "${:,.2f}",
                "Current Price":  "${:,.2f}",
                "Market Value":   "${:,.2f}",
                "Unrealized P/L": "${:,.2f}",
                "P/L %":          "{:+.2f}%",
            })\
            .map(colour_pl, subset=["Unrealized P/L", "P/L %"])
        st.dataframe(styled, width='stretch', hide_index=True)

        # Allocation pie chart
        st.subheader("Capital Allocation")
        alloc_df = df_pos[["Symbol", "Market Value"]].copy()
        alloc_df["Market Value"] = alloc_df["Market Value"].abs()
        fig_pie = px.pie(alloc_df, names="Symbol", values="Market Value",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, width='stretch')

        # Summary metrics
        total_mv  = sum(float(p.market_value)   for p in positions)
        total_upl = sum(float(p.unrealized_pl)  for p in positions)
        m1, m2, m3 = st.columns(3)
        m1.metric("Open Positions",        len(positions))
        m2.metric("Total Market Value",    f"${total_mv:,.2f}")
        m3.metric("Total Unrealized P/L",  f"${total_upl:+,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ORDERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_orders:
    st.subheader("Recent Orders")

    order_filter = st.selectbox("Filter by status", ["all", "filled", "open", "canceled"], index=0)
    try:
        filtered_orders = safe_call(api.list_orders, status=order_filter, limit=50) or []
    except Exception:
        filtered_orders = orders

    if not filtered_orders:
        st.info("No orders found.")
    else:
        rows = []
        for o in filtered_orders:
            rows.append({
                "Symbol":       o.symbol,
                "Side":         o.side.upper(),
                "Qty":          o.qty,
                "Type":         o.order_type,
                "Status":       o.status,
                "Filled Price": f"${float(o.filled_avg_price):.2f}" if o.filled_avg_price else "—",
                "Submitted":    str(o.submitted_at)[:19] if o.submitted_at else "—",
            })
        df_ord = pd.DataFrame(rows)

        def colour_side(val):
            return 'color: #26a69a' if val == 'BUY' else 'color: #ef5350'

        st.dataframe(
            df_ord.style.map(colour_side, subset=["Side"]),
            width='stretch',
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("Risk Monitor")

    # Read the live risk-lever settings directly from backtester config values
    # (imported lazily so the dashboard doesn't need to run the bot)
    try:
        import trading_bot as tb
        risk_rows = [
            ("ATR Position Sizing",  str(tb.USE_ATR_POSITION_SIZING),
             f"Risk per trade: {tb.RISK_PER_TRADE_PCT*100:.1f}% of equity | ATR multiple: {tb.ATR_STOP_MULTIPLE}x"),
            ("ATR Trailing Stops",   str(tb.USE_ATR_STOPS),
             f"Stop distance = ATR x {tb.ATR_STOP_MULTIPLE}  (flat fallback: {tb.TRAILING_STOP_PCT*100:.0f}%)"),
            ("Drawdown Throttle",    str(tb.USE_DRAWDOWN_THROTTLE),
             f"Levels: " + " | ".join(f"{t*100:.0f}% -> {m*100:.0f}% size" for t, m in tb.DRAWDOWN_THROTTLE_LEVELS)),
            ("Take-Profit Target",   "Always on", f"{tb.TAKE_PROFIT_PCT*100:.0f}%"),
            ("Max Simultaneous Pos", "Always on", str(tb.MAX_SIMULTANEOUS_POSITIONS)),
        ]
        df_risk = pd.DataFrame(risk_rows, columns=["Lever", "Status", "Parameters"])

        def colour_status(val):
            if val == "True":  return "color: #26a69a"
            if val == "False": return "color: #ef5350"
            return ""

        st.dataframe(
            df_risk.style.map(colour_status, subset=["Status"]),
            width='stretch', hide_index=True,
        )
    except Exception as e:
        st.warning(f"Could not load bot config: {e}")

    st.divider()

    # Live drawdown status
    st.subheader("Live Drawdown Status")
    if account:
        try:
            raw_hist2 = api.get_portfolio_history(period="90D", timeframe="1D")
            hist2 = pd.DataFrame({"equity": raw_hist2.equity})
            peak  = hist2['equity'].max()
            cur   = float(account.equity)
            dd    = (cur - peak) / peak * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Equity", f"${cur:,.2f}")
            col2.metric("90-Day Peak",    f"${peak:,.2f}")
            col3.metric("Current Drawdown", f"{dd:.2f}%", delta=f"{dd:.2f}%", delta_color="inverse")

            # Visual gauge
            pct_to_10 = min(abs(dd) / 10 * 100, 100)  # progress toward -10% throttle trigger
            if dd < -10:
                st.error("THROTTLE ACTIVE: new positions blocked (>10% drawdown).")
            elif dd < -5:
                st.warning(f"THROTTLE ACTIVE: position sizes halved (>5% drawdown). {pct_to_10:.0f}% toward full stop.")
            else:
                st.success(f"Drawdown OK ({dd:.2f}%). {pct_to_10:.0f}% toward 5% throttle trigger.")
            st.progress(pct_to_10 / 100)
        except Exception as e:
            st.error(f"Could not compute drawdown: {e}")

    st.divider()

    # Open-position ATR / stop levels
    st.subheader("Position Risk Levels (ATR Stops)")
    if not positions:
        st.info("No open positions.")
    else:
        try:
            stop_rows = []
            for p in positions:
                sym     = p.symbol
                qty     = int(float(p.qty))
                entry   = float(p.avg_entry_price)
                current = float(p.current_price)
                # Fetch recent daily bars to compute ATR
                bars = api.get_bars(sym, TimeFrame.Day,
                                    start=(pd.Timestamp.now() - pd.Timedelta(days=30)).date().isoformat(),
                                    end=pd.Timestamp.now().date().isoformat(),
                                    feed='iex', adjustment='split').df
                if not bars.empty and len(bars) >= 14:
                    atr_val = ta.volatility.average_true_range(
                        bars['high'], bars['low'], bars['close'], window=14
                    ).iloc[-1]
                    hwm        = max(entry, current)
                    stop_price = hwm - atr_val * 2.5
                    risk_per_share = atr_val * 2.5
                else:
                    atr_val = stop_price = risk_per_share = None

                stop_rows.append({
                    "Symbol":      sym,
                    "Qty":         qty,
                    "Entry":       entry,
                    "Current":     current,
                    "ATR (14)":    f"${atr_val:.2f}" if atr_val else "N/A",
                    "Stop Price":  f"${stop_price:.2f}" if stop_price else "N/A",
                    "Risk/Share":  f"${risk_per_share:.2f}" if risk_per_share else "N/A",
                    "Max $ Risk":  f"${risk_per_share * abs(qty):.2f}" if risk_per_share else "N/A",
                })
            st.dataframe(pd.DataFrame(stop_rows), width='stretch', hide_index=True)
        except Exception as e:
            st.error(f"Error computing stop levels: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRICE CHART
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.subheader("Price Chart with Indicators")

    col_sym, col_days = st.columns([2, 3])
    chart_symbol = col_sym.text_input("Symbol", "AAPL").upper()
    chart_days   = col_days.slider("Days of history", 30, 365, 120, key="chart_days")

    with st.expander("Indicator settings", expanded=False):
        ic1, ic2, ic3 = st.columns(3)
        sma_p   = ic1.number_input("SMA period",      5, 200, 20)
        ema_p   = ic1.number_input("EMA period",      5, 200, 50)
        rsi_p   = ic2.number_input("RSI period",      5,  50, 14)
        macd_f  = ic3.number_input("MACD fast",       5, 100, 12)
        macd_s  = ic3.number_input("MACD slow",      10, 200, 26)
        macd_sg = ic3.number_input("MACD signal",     3,  50,  9)

    if chart_symbol:
        try:
            end_dt   = pd.Timestamp.now(tz='America/New_York').date()
            start_dt = end_dt - pd.Timedelta(days=chart_days)
            bars = api.get_bars(chart_symbol, TimeFrame.Day,
                                start=start_dt.isoformat(), end=end_dt.isoformat(),
                                feed='iex', adjustment='split').df
            if bars.empty:
                st.info(f"No data for {chart_symbol}.")
            else:
                bars['SMA']  = ta.trend.sma_indicator(bars['close'], window=sma_p)
                bars['EMA']  = ta.trend.ema_indicator(bars['close'], window=ema_p)
                bars['RSI']  = ta.momentum.rsi(bars['close'], window=rsi_p)
                macd_obj     = ta.trend.MACD(bars['close'], window_fast=macd_f,
                                              window_slow=macd_s, window_sign=macd_sg)
                bars['MACD']      = macd_obj.macd()
                bars['MACD_Sig']  = macd_obj.macd_signal()
                bars['MACD_Hist'] = macd_obj.macd_diff()

                fig = go.Figure()

                # Panel 1: candlestick + SMAs
                fig.add_trace(go.Candlestick(
                    x=bars.index, open=bars['open'], high=bars['high'],
                    low=bars['low'], close=bars['close'], name='Price',
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ))
                fig.add_trace(go.Scatter(x=bars.index, y=bars['SMA'],
                    mode='lines', name=f'SMA {sma_p}',
                    line=dict(color='#ffa726', width=1.5)))
                fig.add_trace(go.Scatter(x=bars.index, y=bars['EMA'],
                    mode='lines', name=f'EMA {ema_p}',
                    line=dict(color='#ab47bc', width=1.5)))

                # Panel 2: RSI
                fig.add_trace(go.Scatter(x=bars.index, y=bars['RSI'],
                    mode='lines', name='RSI',
                    line=dict(color='#29b6f6', width=1.5), yaxis='y2'))
                fig.add_hline(y=70, line_dash='dot', line_color='#ef5350',
                              yref='y2', annotation_text='Overbought')
                fig.add_hline(y=30, line_dash='dot', line_color='#26a69a',
                              yref='y2', annotation_text='Oversold')

                # Panel 3: MACD
                fig.add_trace(go.Scatter(x=bars.index, y=bars['MACD'],
                    mode='lines', name='MACD',
                    line=dict(color='#4fc3f7', width=1.5), yaxis='y3'))
                fig.add_trace(go.Scatter(x=bars.index, y=bars['MACD_Sig'],
                    mode='lines', name='Signal',
                    line=dict(color='#ff7043', width=1.2), yaxis='y3'))
                fig.add_trace(go.Bar(x=bars.index, y=bars['MACD_Hist'],
                    name='Histogram', yaxis='y3',
                    marker_color=['#26a69a' if v >= 0 else '#ef5350'
                                  for v in bars['MACD_Hist']]))

                fig.update_layout(
                    title=f"{chart_symbol} — {chart_days}-day chart",
                    height=750,
                    margin=dict(t=40, b=20, l=10, r=10),
                    xaxis=dict(rangeslider_visible=False, domain=[0, 1]),
                    yaxis =dict(domain=[0.55, 1.00], title='Price ($)'),
                    yaxis2=dict(domain=[0.28, 0.50], title='RSI'),
                    yaxis3=dict(domain=[0.00, 0.24], title='MACD'),
                    hovermode='x unified',
                    legend=dict(orientation='h', y=1.03),
                )
                st.plotly_chart(fig, width='stretch')
        except Exception as e:
            st.error(f"Chart error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MANUAL ORDER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Place a Manual Order")
    st.info("This places a real paper-trading order via Alpaca immediately on submit.")

    c1, c2, c3 = st.columns(3)
    o_sym  = c1.text_input("Symbol", "").upper()
    o_qty  = c2.number_input("Quantity", min_value=1, value=1, step=1)
    o_side = c3.radio("Side", ("Buy", "Sell"))

    if st.button("Submit Order", type="primary"):
        if not o_sym:
            st.error("Symbol is required.")
        else:
            try:
                result = api.submit_order(
                    symbol=o_sym, qty=o_qty,
                    side="buy" if o_side == "Buy" else "sell",
                    type="market", time_in_force="day",
                    client_order_id=str(uuid.uuid4()),
                )
                st.success(f"Order submitted: {o_side} {o_qty} x {o_sym}")
                st.json(result._raw)
            except Exception as e:
                st.error(f"Order failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — BOT LOGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("Live Bot Log")

    log_lines = st.slider("Lines to show", 20, 200, 60)
    log_file  = st.text_input("Log file path", "bot_log.txt")

    def read_log(path, n):
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            return "".join(lines[-n:])
        except FileNotFoundError:
            return f"Log file not found: {path}\nStart the bot with:\n  nohup python trading_bot.py > bot_log.txt 2>&1 &"
        except Exception as e:
            return f"Error reading log: {e}"

    log_placeholder = st.empty()
    refresh_log = st.button("Refresh log")

    log_content = read_log(log_file, log_lines)
    log_placeholder.code(log_content, language="text")

    if refresh_log:
        log_content = read_log(log_file, log_lines)
        log_placeholder.code(log_content, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
WF_RESULTS_PATH = "/tmp/wf_results.json"
WF_PID_PATH     = "/tmp/wf_running.pid"

def _wf_is_running():
    """Return True if a walk_forward.py subprocess is currently active."""
    if not os.path.exists(WF_PID_PATH):
        return False
    try:
        pid = int(open(WF_PID_PATH).read().strip())
        os.kill(pid, 0)   # signal 0 = "does this process exist?"
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False

def _wf_load_results():
    """Load saved JSON results, or return None if the file doesn't exist."""
    if not os.path.exists(WF_RESULTS_PATH):
        return None
    try:
        with open(WF_RESULTS_PATH) as f:
            return json.load(f)
    except Exception:
        return None

with tab_wf:
    st.subheader("Walk-Forward Validation")
    st.caption(
        "Tests whether the strategy's historical edge is consistent across "
        "independent calendar years — the primary check against overfitting."
    )

    # ── Run / status controls ─────────────────────────────────────────────────
    col_btn, col_status = st.columns([2, 5])

    with col_btn:
        run_clicked = st.button("Run Walk-Forward Analysis", type="primary", width='stretch')
        if run_clicked:
            if _wf_is_running():
                st.warning("Analysis already running.")
            else:
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walk_forward.py")
                proc = subprocess.Popen(
                    [sys.executable, script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with open(WF_PID_PATH, "w") as f:
                    f.write(str(proc.pid))
                st.info(f"Analysis started (PID {proc.pid}). This takes several minutes. "
                        "Return to this tab and click Refresh when done.")

        if st.button("Refresh Results", width='stretch'):
            st.rerun()

    with col_status:
        if _wf_is_running():
            st.warning("Walk-forward analysis is currently running... refresh this tab to check for completion.")

    st.divider()

    # ── Load and display results ──────────────────────────────────────────────
    wf_data = _wf_load_results()

    if wf_data is None:
        st.info(
            "No results found. Click **Run Walk-Forward Analysis** above to generate them.  \n"
            "The analysis fetches 3 years of data and simulates all strategies — expect 10-20 minutes."
        )
    else:
        gen_at = wf_data.get("generated_at", "unknown")
        st.caption(f"Results generated: {gen_at}")

        windows = wf_data.get("windows", [])
        if not windows:
            st.warning("Results file exists but contains no window data.")
            st.stop()

        labels      = [w["label"] for w in windows]
        pl_pcts     = [w["pl_pct"] for w in windows]
        sharpes     = [w["sharpe"] for w in windows]
        drawdowns   = [w["max_drawdown"] for w in windows]
        num_trades  = [w["num_trades"] for w in windows]
        final_eqs   = [w["final_equity"] for w in windows]

        # ── Summary metric cards ──────────────────────────────────────────────
        cols = st.columns(len(windows))
        for col, w in zip(cols, windows):
            sharpe_str = f"{w['sharpe']:.2f}" if w['sharpe'] is not None else "N/A"
            col.metric(
                label=f"{w['label']}  |  {w['start_date']} – {w['end_date']}",
                value=f"{w['pl_pct']:+.2f}%",
                delta=f"Sharpe {sharpe_str}",
                delta_color="normal",
            )

        st.divider()

        # ── Chart 1: Annual returns bar chart ─────────────────────────────────
        # Each bar is one calendar year. Green = positive, red = negative.
        # Shows the raw return % and annotates the $ P/L on each bar.
        # SPY buy-and-hold markers give the all-important benchmark context:
        # a green bar BELOW its SPY marker still underperformed the index.
        benchmarks = [w.get("benchmark_pct") for w in windows]
        fig_ret = go.Figure()
        bar_colors = ['#26a69a' if p >= 0 else '#ef5350' for p in pl_pcts]
        fig_ret.add_trace(go.Bar(
            x=labels,
            y=pl_pcts,
            marker_color=bar_colors,
            text=[f"${w['total_pl']:+,.0f}" for w in windows],
            textposition='outside',
            name='Strategy Return %',
        ))
        if any(b is not None for b in benchmarks):
            fig_ret.add_trace(go.Scatter(
                x=labels,
                y=benchmarks,
                mode='markers',
                name='SPY Buy-and-Hold',
                marker=dict(symbol='line-ew-open', size=38, color='#ffb74d',
                            line=dict(width=3, color='#ffb74d')),
            ))
        fig_ret.update_layout(
            title="Annual Return by Year (vs SPY buy-and-hold)",
            height=350,
            yaxis=dict(title="Return (%)", ticksuffix="%", zeroline=True, zerolinecolor="#555"),
            xaxis=dict(title="Year"),
            margin=dict(t=50, b=30, l=10, r=10),
            legend=dict(orientation='h', y=1.08),
        )
        st.plotly_chart(fig_ret, width='stretch')

        # ── Chart 2: Normalised equity curves ────────────────────────────────
        # Normalise each year's equity to start at 100 so the curves are
        # visually comparable regardless of starting capital.
        # This is the standard quant way to compare periods: you care about
        # the SHAPE (drawdown depth, recovery speed) not the absolute level.
        colors = ['#4fc3f7', '#ffb74d', '#a5d6a7', '#ce93d8']
        fig_eq = go.Figure()
        for i, w in enumerate(windows):
            curve = w.get("daily_equity_curve", [])
            if not curve:
                continue
            df_c = pd.DataFrame(curve)
            df_c['date'] = pd.to_datetime(df_c['date'])
            df_c = df_c.set_index('date').sort_index()
            base = df_c['equity'].iloc[0]
            df_c['normalised'] = df_c['equity'] / base * 100
            fig_eq.add_trace(go.Scatter(
                x=df_c.index,
                y=df_c['normalised'],
                mode='lines',
                name=w['label'],
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_eq.add_hline(y=100, line_dash="dash", line_color="#555", annotation_text="Start")
        fig_eq.update_layout(
            title="Normalised Equity Curves (start = 100)",
            height=380,
            yaxis=dict(title="Indexed Equity"),
            xaxis=dict(title="Date", rangeslider_visible=False),
            hovermode='x unified',
            margin=dict(t=50, b=30, l=10, r=10),
            legend=dict(orientation='h', y=1.05),
        )
        st.plotly_chart(fig_eq, width='stretch')

        # ── Chart 3: Drawdown + Sharpe side by side ───────────────────────────
        c_dd, c_sh = st.columns(2)

        with c_dd:
            fig_dd = go.Figure(go.Bar(
                x=labels,
                y=drawdowns,
                marker_color='#ef5350',
                text=[f"{d:.1f}%" for d in drawdowns],
                textposition='outside',
            ))
            fig_dd.update_layout(
                title="Max Drawdown by Year",
                height=300,
                yaxis=dict(title="Drawdown (%)", ticksuffix="%"),
                margin=dict(t=50, b=30, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_dd, width='stretch')

        with c_sh:
            sh_vals = [s if s is not None else 0 for s in sharpes]
            sh_colors = ['#26a69a' if s >= 0.5 else '#ffb74d' if s >= 0 else '#ef5350' for s in sh_vals]
            fig_sh = go.Figure(go.Bar(
                x=labels,
                y=sh_vals,
                marker_color=sh_colors,
                text=[f"{s:.2f}" if s is not None else "N/A" for s in sharpes],
                textposition='outside',
            ))
            fig_sh.add_hline(y=1.0, line_dash="dash", line_color="#888",
                             annotation_text="Sharpe 1.0 threshold")
            fig_sh.update_layout(
                title="Annualised Sharpe Ratio by Year",
                height=300,
                yaxis=dict(title="Sharpe Ratio"),
                margin=dict(t=50, b=30, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_sh, width='stretch')

        # ── Chart 4: Per-strategy P/L grouped by year ────────────────────────
        # Grouped bar: each cluster = one strategy; bars within the cluster =
        # the years.  Lets you see whether a strategy is consistently profitable
        # or only worked in a specific market environment.
        st.subheader("Strategy P/L Breakdown by Year")

        # Build a unified DataFrame: rows = (strategy, year), col = total_pl
        rows = []
        for w in windows:
            for s in w.get("strategy_breakdown", []):
                rows.append({
                    'year':     w['label'],
                    'strategy': s['strategy'],
                    'total_pl': s['total_pl'],
                    'win_rate': s['win_rate'],
                    'trades':   s['trades'],
                })
        if rows:
            bkdn_df = pd.DataFrame(rows)
            all_strategies = bkdn_df['strategy'].unique().tolist()

            fig_strat = go.Figure()
            year_colors = {'2023': '#4fc3f7', '2024': '#ffb74d', '2025': '#a5d6a7'}
            for yr in labels:
                yr_df = bkdn_df[bkdn_df['year'] == yr]
                fig_strat.add_trace(go.Bar(
                    name=yr,
                    x=yr_df['strategy'],
                    y=yr_df['total_pl'],
                    marker_color=year_colors.get(yr, '#ce93d8'),
                    text=[f"${v:,.0f}" for v in yr_df['total_pl']],
                    textposition='outside',
                    customdata=yr_df[['win_rate', 'trades']].values,
                    hovertemplate=(
                        "<b>%{x}</b> — %{meta}<br>"
                        "P/L: $%{y:,.0f}<br>"
                        "Win rate: %{customdata[0]:.1f}%<br>"
                        "Trades: %{customdata[1]}<extra></extra>"
                    ),
                    meta=yr,
                ))
            fig_strat.add_hline(y=0, line_color="#555")
            fig_strat.update_layout(
                barmode='group',
                height=400,
                yaxis=dict(title="Total P/L ($)", tickprefix="$"),
                xaxis=dict(title="Strategy"),
                hovermode='closest',
                legend=dict(title="Year", orientation='h', y=1.05),
                margin=dict(t=50, b=30, l=10, r=10),
            )
            st.plotly_chart(fig_strat, width='stretch')
        else:
            st.info("No per-strategy breakdown data available in results.")

        # ── Verdict summary ───────────────────────────────────────────────────
        st.divider()
        positive_years = sum(1 for p in pl_pcts if p and p > 0)
        valid_sharpes  = [s for s in sharpes if s is not None]
        avg_sharpe     = sum(valid_sharpes) / len(valid_sharpes) if valid_sharpes else 0

        verdict_col, table_col = st.columns([3, 2])
        with verdict_col:
            st.subheader("Verdict")
            if positive_years == len(windows) and avg_sharpe >= 1.0:
                st.success(f"{positive_years}/{len(windows)} years positive | Avg Sharpe {avg_sharpe:.2f} — strong, consistent edge.")
            elif positive_years >= 2 and avg_sharpe >= 0.3:
                st.warning(f"{positive_years}/{len(windows)} years positive | Avg Sharpe {avg_sharpe:.2f} — edge present but not uniform.")
            else:
                st.error(f"{positive_years}/{len(windows)} years positive | Avg Sharpe {avg_sharpe:.2f} — strategy may be overfitted or structurally weak.")

        with table_col:
            summary_df = pd.DataFrame({
                "Year":          labels,
                "Return":        [f"{p:+.2f}%" for p in pl_pcts],
                "SPY":           [f"{w.get('benchmark_pct'):+.2f}%" if w.get('benchmark_pct') is not None else "N/A" for w in windows],
                "Alpha":         [f"{w.get('alpha_pp'):+.2f}pp" if w.get('alpha_pp') is not None else "N/A" for w in windows],
                "Sharpe":        [f"{s:.2f}" if s is not None else "N/A" for s in sharpes],
                "Max DD":        [f"{d:.2f}%" for d in drawdowns],
                "Trades":        num_trades,
                "Final Equity":  [f"${e:,.0f}" for e in final_eqs],
            })
            st.dataframe(summary_df, hide_index=True, width='stretch')

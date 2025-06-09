import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from config import DATA_DIR, PLOT_DIR, KLINE_DAYS


def load_trade_details(path):
    """Load trade details CSV and append numeric return_pct column."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["return_pct"] = df["return"].str.rstrip("%").astype(float) / 100
    df["buy_date"] = pd.to_datetime(df["buy_date"])
    return df


def plot_return_pie(df, outfile):
    """Plot win/loss pie chart."""
    wins = (df["return_pct"] > 0).sum()
    losses = (df["return_pct"] <= 0).sum()
    plt.figure()
    plt.pie([wins, losses], labels=["Win", "Loss"], autopct="%1.1f%%")
    plt.title("Win vs Loss")
    plt.savefig(outfile)
    plt.close()


def plot_return_line(df, outfile):
    """Plot cumulative return line chart."""
    cum = (1 + df["return_pct"]).cumprod()
    plt.figure()
    plt.plot(df["buy_date"], cum)
    plt.xlabel("Buy Date")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Returns")
    plt.grid(True)
    plt.savefig(outfile)
    plt.close()


def plot_kline(data_file, buy_date, n_days, outfile):
    """Plot candlestick chart around the buy date."""
    df = pd.read_csv(data_file, parse_dates=["trade_date"], encoding="utf-8-sig")
    df.rename(columns={"trade_date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "vol": "Volume"}, inplace=True)
    df.set_index("Date", inplace=True)

    buy_date = pd.to_datetime(buy_date)
    start = buy_date - pd.Timedelta(days=n_days)
    end = buy_date + pd.Timedelta(days=n_days)
    sub = df.loc[start:end]

    addplots = []
    if buy_date in sub.index:
        scatter = mpf.make_addplot(pd.Series(sub.loc[buy_date, "Close"], index=[buy_date]),
                                   type="scatter", markersize=100, marker="^", color="r")
        addplots.append(scatter)

    mpf.plot(sub, type="candle", style="charles", addplot=addplots, volume=True,
             savefig=outfile)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    detail_files = glob.glob(os.path.join("results", "*_details.csv"))
    if not detail_files:
        print("No _details.csv files found in results directory")
        return

    for path in detail_files:
        df = load_trade_details(path)
        stock_full = os.path.basename(path).replace("_details.csv", "")
        stock_code = stock_full.split("_")[0]  # 只保留代码部分

        out_dir = os.path.join(PLOT_DIR, stock_code)
        os.makedirs(out_dir, exist_ok=True)

        plot_return_pie(df, os.path.join(out_dir, "pie.png"))
        plot_return_line(df, os.path.join(out_dir, "line.png"))

        data_file = os.path.join(DATA_DIR, f"{stock_code}.csv")
        if not os.path.exists(data_file):
            print(f"Data file missing: {data_file}")
            continue

        for _, row in df.iterrows():
            buy = row["buy_date"].strftime("%Y-%m-%d")
            outfile = os.path.join(out_dir, f"kline_{buy}.png")
            plot_kline(data_file, buy, KLINE_DAYS, outfile)


if __name__ == "__main__":
    main()
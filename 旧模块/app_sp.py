import pandas as pd
import plotly.express as px
import gradio as gr
import os
# —— 强制本地不走代理 ——
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")


PROB_CSV = "./output/stats_prelaunch/prob_bins.csv"
HDI_CSV = "./output/stats_prelaunch/stats_hdi.csv"

def load_data():
    try:
        prob_df = pd.read_csv(PROB_CSV)
    except Exception:
        prob_df = pd.DataFrame(columns=["feature","rel_day","bin","count","prob"])
    try:
        hdi_df = pd.read_csv(HDI_CSV)
    except Exception:
        hdi_df = pd.DataFrame(columns=["feature","rel_day","count","mean","std","median",
                                       "hdi_low","hdi_high","hdi_width","hdi_k"])
    return prob_df, hdi_df

prob_df, hdi_df = load_data()
features = sorted(prob_df["feature"].dropna().unique().tolist()) if not prob_df.empty else []
rel_days_all = sorted(prob_df["rel_day"].dropna().unique().tolist()) if not prob_df.empty else []

def plot_feature(feature, rel_days):
    if prob_df.empty or not feature:
        return px.bar(title="没有可用的 prob_bins 数据")
    sub = prob_df[prob_df["feature"] == feature].copy()
    if rel_days:
        sub = sub[sub["rel_day"].isin(rel_days)]
    if not sub.empty:
        base_day = rel_days[0] if rel_days else sub["rel_day"].iloc[0]
        order = sub[sub["rel_day"] == base_day]["bin"].astype(str).unique().tolist()
        sub["bin"] = pd.Categorical(sub["bin"].astype(str), categories=order, ordered=True)
    fig = px.bar(sub, x="bin", y="prob", color="rel_day", barmode="group",
                 title=f"{feature} 分箱概率分布",
                 labels={"bin":"分箱区间","prob":"概率（占比）","rel_day":"相对天数"})
    return fig

with gr.Blocks() as demo:
    gr.Markdown("# 预启动特征 · 分箱概率可视化")
    with gr.Row():
        feature_dd = gr.Dropdown(choices=features, label="选择指标", value=(features[0] if features else None))
        rel_dd = gr.CheckboxGroup(choices=rel_days_all, label="选择相对天数", value=(rel_days_all[:1] if rel_days_all else []))
    plot = gr.Plot()
    btn = gr.Button("绘制 / 刷新")
    btn.click(plot_feature, inputs=[feature_dd, rel_dd], outputs=plot)

    gr.Markdown("## HDI 统计预览（只读）")
    gr.Dataframe(value=hdi_df, interactive=False, wrap=True, )

    # 初始自动渲染
    if features:
        plot.value = plot_feature(features[0], rel_days_all[:1])

demo.launch(
        share=False,                 # 强制本地，不去申请外网隧道
        server_name="127.0.0.1",     # 只监听回环地址
        server_port=None,            # 端口冲突时自动换
        inbrowser=True,              # 自动在浏览器打开
        show_error=True
    )

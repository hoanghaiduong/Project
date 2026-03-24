import os
import pandas as pd
import streamlit as st

from src.data_loader import load_elliptic_pyg_data


st.set_page_config(page_title="AML Blockchain Demo", layout="wide")
st.title("Demo phát hiện giao dịch rửa tiền AML trên Elliptic")
st.markdown("Ứng dụng demo cho môn Lý thuyết đồ thị và ứng dụng.")

with st.sidebar:
    st.header("Cấu hình")
    data_dir = st.text_input("Data directory", value="data")
    demo = st.checkbox("Demo mode (nhanh)", value=True)
    demo_nodes = st.slider("Demo nodes", min_value=1000, max_value=10000, value=4000, step=500)

if st.button("Tải dữ liệu và hiển thị thống kê"):
    try:
        data, _, _, stats = load_elliptic_pyg_data(data_dir, demo=demo, demo_nodes=demo_nodes)
        st.success("Tải dữ liệu thành công")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Nodes", stats["num_nodes"])
        c2.metric("Edges", stats["num_edges"])
        c3.metric("Licit", stats["num_licit"])
        c4.metric("Illicit", stats["num_illicit"])
        c5.metric("Features", stats["num_features"])
    except Exception as e:
        st.error(f"Không thể đọc dữ liệu: {e}")

st.header("Kết quả đã huấn luyện")
report_path = "outputs/reports/model_comparison.csv"
if os.path.exists(report_path):
    df = pd.read_csv(report_path)
    st.dataframe(df, use_container_width=True)
else:
    st.info("Chưa có model_comparison.csv. Hãy chạy main.py trước.")

st.header("Biểu đồ")
fig_map = {
    "Loss & Accuracy": "outputs/figures/loss_accuracy_curves.png",
    "Label Distribution": "outputs/figures/label_distribution.png",
    "Degree Distribution": "outputs/figures/degree_distribution.png",
    "Timestep Distribution": "outputs/figures/timestep_distribution.png",
    "Feature Correlation": "outputs/figures/feature_correlation_heatmap.png",
    "Subgraph": "outputs/figures/subgraph_visualization.png",
    "Attention Heatmap": "outputs/figures/attention_heatmap.png",
    "CM Graph Transformer": "outputs/figures/cm_graph_transformer.png",
    "CM GCN": "outputs/figures/cm_gcn.png",
    "CM ANN": "outputs/figures/cm_ann.png",
    "CM XGBoost": "outputs/figures/cm_xgboost.png",
}

for title, path in fig_map.items():
    st.subheader(title)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.caption(f"Chưa có file: {path}")

st.markdown("---")
st.caption("Chạy app: streamlit run app.py")

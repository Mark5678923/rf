import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 页面设置
st.set_page_config(page_title="Mortality Risk Predictor", layout="centered")

# 自定义 CSS 样式
st.markdown("""
    <style>
        html, body {
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            font-size: 28px !important;
            text-align: center;
            color: #2c3e50;
        }
        h2, h3 {
            font-size: 20px !important;
            color: #34495e;
        }
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton button:hover {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# 标题
st.title("T2DM-CHD All-Cause Mortality Risk Predictor")

# 加载模型
model = joblib.load('rf.pkl')

# 特征定义
feature_ranges = {
    "CRP": {"type": "numerical", "min": 0.00, "max": 15.00, "default": 0.54, "unit": "mg/dL"},
    "eGFR": {"type": "numerical", "min": 3.00, "max": 180.00, "default": 114.78, "unit": "mL/min/1.73m²"},
    "TyG_BMI": {"type": "numerical", "min": 100.00, "max": 800.00, "default": 394.20, "unit": ""},
    "Age": {"type": "numerical", "min": 20, "max": 120, "default": 54, "unit": "years"},
    "PIR": {"type": "numerical", "min": 0.00, "max": 5.00, "default": 1.84, "unit": ""},
    "MCV": {"type": "numerical", "min": 50.0, "max": 150.0, "default": 92.9, "unit": "fL"},
    "ALP": {"type": "numerical", "min": 0.0, "max": 700.0, "default": 93.0, "unit": "U/L"},
    "PLT": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 192.0, "unit": "*10⁹/L"}
}

# 输入区域
st.markdown("### 🧪 Please input the patient's clinical features:")
feature_values = []
cols = st.columns(2)
for idx, (feature, props) in enumerate(feature_ranges.items()):
    unit = f" ({props['unit']})" if props.get("unit") else ""
    label = f"{feature}{unit} [{props['min']} - {props['max']}]"
    with cols[idx % 2]:
        value = st.number_input(
            label=label,
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
            key=feature
        )
        feature_values.append(value)

# 转换为 DataFrame
features = np.array([feature_values])
df_input = pd.DataFrame(features, columns=feature_ranges.keys())

# 预测按钮
if st.button("🔍 Predict Risk"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    class_names = {0: "🟢 Low Risk (Survival)", 1: "🔴 High Risk (Mortality)"}
    class_name = class_names[predicted_class]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果
    st.markdown("### 📊 Prediction Result")
    st.success(f"**{class_name}**\n\nProbability: **{probability:.2f}%**")

    # 解释部分
    st.markdown("### 🧠 SHAP Feature Contribution")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

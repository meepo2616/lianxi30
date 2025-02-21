import streamlit as st
import joblib
import numpy as np
import pandas as pd


# 加载保存的模型
model = joblib.load('XGB.pkl')

# 加载模型时添加异常捕获
try:
    model = joblib.load('XGB.pkl')
    selected_features = joblib.load('selected_features.pkl')  # 加载前4个特征名称
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# 特征范围定义（根据提供的特征范围和数据类型）
full_feature_ranges = {
    "Vw": {"type": "numerical", "min": 0, "max": 1000000, "default": 10400},
    "Bave": {"type": "numerical", "min": 0, "max": 100, "default": 50},
    "hd": {"type": "numerical", "min": 0, "max": 100, "default": 50},
    "hb": {"type": "numerical", "min": 0, "max": 100, "default": 50},
    "S": {"type": "numerical", "min": 0, "max": 1000000, "default": 10400},
    "hw": {"type": "numerical", "min": 0, "max": 100, "default": 50},
}

# Streamlit 界面
st.title("Qp Prediction with XGBoost")

# 动态生成输入项（仅显示前4个重要特征）
st.header("Input Feature Values")
feature_values = []
for feature in selected_features:  # 只遍历选中的特征
    props = full_feature_ranges[feature]
    value = st.number_input(
        f"{feature} ({props['min']} - {props['max']})",
        min_value=props["min"],
        max_value=props["max"],
        value=props["default"],
    )
    feature_values[feature] = value

# 预测逻辑
if st.button("Predict Qp"):
    try:
        # 确保输入顺序与模型训练时一致
        input_data = pd.DataFrame([feature_values], columns=selected_features)
        prediction = model.predict(input_data)[0]
        st.success(f"**Predicted Qp:** {prediction:.2f}")


    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")



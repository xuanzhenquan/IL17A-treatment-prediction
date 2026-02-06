import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =========================================================
# 1. 页面配置与模型加载
# =========================================================
st.set_page_config(page_title="IL-17A 疗效预测", layout="centered")

# 加载保存的模型
try:
    # 确保你的模型文件名叫 rf.pkl
    model = joblib.load('rf_model.pkl') 
except FileNotFoundError:
    st.error("❌ 未找到模型文件 'rf.pkl'。请确保文件在同一目录下。")
    st.stop()

# =========================================================
# 2. 定义特征范围 (精准适配你的7个变量)
# =========================================================
# ⚠️ 关键：字典的顺序必须与你训练模型时 X_train 的列顺序保持一致！
# 如果顺序不对，预测结果会完全错误。请核对下方顺序。

feature_ranges = {
    # 1. BMI (体重指数)
    "BMI": {
        "type": "numerical", 
        "min": 10.0, "max": 50.0, "default": 24.0, 
        "label": "体重指数 (BMI)"
    },
    
    # 2. Biologics_History (既往生物制剂史) - 假设 0=无, 1=有
    "Biologics_History": {
        "type": "categorical", 
        "options": [0, 1], "default": 0, 
        "label": "既往生物制剂使用史 (0=无, 1=有)"
    },
    
    # 3. Baseline_PASI (基线 PASI 评分)
    "Baseline_PASI": {
        "type": "numerical", 
        "min": 0.0, "max": 72.0, "default": 15.0, 
        "label": "基线 PASI 评分"
    },
    
    # 4. Hemoglobin (血红蛋白) - 单位通常是 g/L
    "Hemoglobin": {
        "type": "numerical", 
        "min": 50.0, "max": 200.0, "default": 130.0, 
        "label": "血红蛋白 (Hb, g/L)"
    },
    
    # 5. ALP (碱性磷酸酶) - 单位 U/L
    "ALP": {
        "type": "numerical", 
        "min": 10.0, "max": 300.0, "default": 70.0, 
        "label": "碱性磷酸酶 (ALP, U/L)"
    },
    
    # 6. IBil (间接胆红素) - 单位 μmol/L
    "IBil": {
        "type": "numerical", 
        "min": 0.0, "max": 50.0, "default": 10.0, 
        "label": "间接胆红素 (IBil, μmol/L)"
    },
    
    # 7. SII (系统免疫炎症指数) - 这是一个计算值，范围很大
    "SII": {
        "type": "numerical", 
        "min": 0.0, "max": 5000.0, "default": 500.0, 
        "label": "系统免疫炎症指数 (SII)"
    }
}

# =========================================================
# 3. Streamlit 界面：侧边栏输入
# =========================================================
st.title("🏥 IL-17A 抑制剂疗效预测系统")
st.markdown("### 基于 7 个关键特征的随机森林模型")

st.sidebar.header("📋 患者临床指标录入")
st.sidebar.info("请在下方输入患者的 7 个关键指标")

user_inputs = {}

# 循环生成输入框
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=properties["label"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            key=feature
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=properties["label"],
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
            key=feature
        )
    user_inputs[feature] = value

# 转换为 DataFrame
input_df = pd.DataFrame([user_inputs])

# 显示当前输入
st.subheader("1. 患者信息确认")
st.dataframe(input_df)

# =========================================================
# 4. 预测与 SHAP 可视化
# =========================================================
if st.button("🚀 开始预测 (Predict)"):
    st.subheader("2. 预测结果")
    
    # --- 步骤 A: 模型预测 ---
    try:
        # Pipeline 自动处理归一化
        predicted_proba = model.predict_proba(input_df)[0]
        probability_responder = predicted_proba[1] * 100  # Class 1 (有效) 的概率
        
        # 结果文案逻辑
        if probability_responder > 50:
            result_text = "Responder (有效)"
            color_code = "#2ca02c" # 绿色
            advice = "该患者对 IL-17A 治疗反应良好的可能性较高。"
        else:
            result_text = "Non-Responder (无效)"
            color_code = "#d62728" # 红色
            advice = "该患者可能对治疗反应不佳，建议关注风险因素。"

        # --- 步骤 B: 绘制文字结果图 ---
        text = f"Predicted Probability: {probability_responder:.2f}%\nResult: {result_text}"
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, text, fontsize=18, ha='center', va='center',
                fontname='Times New Roman', fontweight='bold', color='black',
                transform=ax.transAxes)
        
        # 边框变色
        for spine in ax.spines.values():
            spine.set_edgecolor(color_code)
            spine.set_linewidth(3)
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        st.info(f"💡 AI 建议：{advice}")

        # --- 步骤 C: SHAP 可视化 ---
        st.subheader("3. AI 决策解释 (SHAP Force Plot)")
        with st.spinner('正在计算特征贡献度...'):
            # 1. 提取组件
            rf_classifier = model.named_steps['classifier']
            scaler = model.named_steps['scaler']
            
            # 2. 归一化输入数据
            input_scaled = scaler.transform(input_df)
            
            # 3. 创建解释器
            explainer = shap.TreeExplainer(rf_classifier)
            shap_values_raw = explainer.shap_values(input_scaled, check_additivity=False)
            
            # 4. 提取 Class 1 的 SHAP 值
            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]
                base_value = explainer.expected_value[1]
            else:
                shap_values = shap_values_raw[:,:,1]
                base_value = explainer.expected_value[1]

            # 5. 绘图
            plt.figure(figsize=(12, 4), dpi=150)
            shap.force_plot(
                base_value,
                shap_values[0],
                input_df.iloc[0],
                feature_names=input_df.columns,
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            st.pyplot(plt)
            st.caption("注：红色条推高有效概率，蓝色条拉低有效概率。")
            
    except Exception as e:
        st.error(f"发生错误: {e}")
        st.warning("请检查：\n1. `feature_ranges` 中的变量名是否与训练时的列名完全一致（区分大小写）。\n2. 变量的顺序是否一致。")

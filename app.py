import streamlit as st
import json
import os
from src import SJTRunner, DataLoader
import pandas as pd

# 常量配置
DEFAULT_N_ITEM = 4
DEFAULT_SCALE = "NEO-PI-R"
DEFAULT_TRAITS = ["O4", "C4", "E5", "A5", "N5"]
DEFAULT_SITUATION_THEME = "大学校园里的日常生活"
DEFAULT_RESULT_DIR = 'output'
DEFAULT_RESULT_SJT_FN = 'sjt-text'
DEFAULT_RESULT_DETAILED_SJT_FN = 'sjt-text_detailed'
DEFAULT_MODEL = 'gpt-5'

# 页面配置
st.set_page_config(
    page_title="SJT生成器",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .trait-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .trait-title {
        font-weight: bold;
        color: #2c3e50;
        font-size: 1.1rem;
    }
    .trait-description {
        color: #34495e;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .config-section {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">🧠 SJT情境判断测验生成器</h1>', unsafe_allow_html=True)

# 初始化数据加载器
@st.cache_data
def load_data():
    try:
        data_loader = DataLoader()
        neopir = data_loader.load(DEFAULT_SCALE, 'zh')
        neopir_meta = data_loader.load_meta(DEFAULT_SCALE)
        return neopir, neopir_meta, data_loader
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return None, None, None

# 侧边栏配置
st.sidebar.markdown("## ⚙️ 配置参数")

# 基本配置
st.sidebar.markdown("### 基础设置")
n_item = st.sidebar.number_input("每个特质生成题目数量", min_value=1, max_value=10, value=DEFAULT_N_ITEM)
situation_theme = st.sidebar.text_input("情境主题", value=DEFAULT_SITUATION_THEME)
model = st.sidebar.selectbox("选择语言模型", ["gpt-5", "gpt-5-mini", "gpt-4"], index=0)

# 文件保存设置
st.sidebar.markdown("### 文件保存设置")
result_dir = st.sidebar.text_input("结果保存目录", value=DEFAULT_RESULT_DIR)
result_sjt_fn = st.sidebar.text_input("SJT结果文件名", value=DEFAULT_RESULT_SJT_FN)
result_detailed_sjt_fn = st.sidebar.text_input("详细结果文件名", value=DEFAULT_RESULT_DETAILED_SJT_FN)

# 加载数据
neopir, neopir_meta, data_loader = load_data()

if neopir and neopir_meta:
    # 特质选择区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">🎯 选择人格特质</div>', unsafe_allow_html=True)
        
        available_traits = list(neopir.keys())
        trait_options = {trait: f"{trait} - {neopir[trait]['facet_name']}" for trait in available_traits}
        
        selected_traits = st.multiselect(
            "请选择要生成SJT题目的人格特质",
            options=available_traits,
            default=DEFAULT_TRAITS,
            format_func=lambda x: trait_options[x]
        )
        
        if selected_traits:
            st.success(f"已选择 {len(selected_traits)} 个特质，将生成 {len(selected_traits) * n_item} 道题目")
    
    with col2:
        st.markdown('<div class="section-header">📊 生成统计</div>', unsafe_allow_html=True)
        st.metric("选中特质数量", len(selected_traits) if selected_traits else 0)
        st.metric("预计题目总数", len(selected_traits) * n_item if selected_traits else 0)
    
    # NEO-PI-R 知识展示区域
    st.markdown('<div class="section-header">📚 NEO-PI-R 人格特质知识库</div>', unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🔍 特质详情", "📋 完整列表", "🎯 已选特质"])
    
    with tab1:
        if neopir_meta:
            search_trait = st.selectbox(
                "选择查看特质详情",
                options=list(neopir_meta.keys()),
                format_func=lambda x: f"{x} - {neopir_meta[x]['facet_name']} ({neopir_meta[x]['domain']})"
            )
            
            if search_trait and search_trait in neopir_meta:
                trait_info = neopir_meta[search_trait]
                st.markdown(f"""
                <div class="trait-card">
                    <div class="trait-title">{search_trait}: {trait_info['facet_name']} ({trait_info['domain']})</div>
                    <div class="trait-description">
                        <strong>描述:</strong> {trait_info['description']}<br>
                        <strong>高分特征:</strong> {trait_info['high_score']}<br>
                        <strong>低分特征:</strong> {trait_info['low_score']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        if neopir_meta:
            domains = {}
            for trait_id, trait_info in neopir_meta.items():
                domain = trait_info['domain']
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append((trait_id, trait_info))
            
            for domain, traits in domains.items():
                st.markdown(f"**{domain} (神经质/外向性/开放性/宜人性/尽责性)**")
                for trait_id, trait_info in traits:
                    with st.expander(f"{trait_id}: {trait_info['facet_name']}"):
                        st.write(f"**描述:** {trait_info['description']}")
                        st.write(f"**高分:** {trait_info['high_score']}")
                        st.write(f"**低分:** {trait_info['low_score']}")
    
    with tab3:
        if selected_traits and neopir_meta:
            for trait in selected_traits:
                if trait in neopir_meta:
                    trait_info = neopir_meta[trait]
                    st.markdown(f"""
                    <div class="trait-card">
                        <div class="trait-title">{trait}: {trait_info['facet_name']} ({trait_info['domain']})</div>
                        <div class="trait-description">{trait_info['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # 生成按钮和进度区域
    st.markdown('<div class="section-header">🚀 生成SJT题目</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 开始生成SJT题目", type="primary", use_container_width=True):
            if not selected_traits:
                st.error("请至少选择一个人格特质！")
            else:
                with st.spinner("正在生成SJT题目，请耐心等待..."):
                    try:
                        # 验证特质可用性
                        available_traits = list(neopir.keys())
                        invalid_traits = [trait for trait in selected_traits if trait not in available_traits]
                        
                        if invalid_traits:
                            st.error(f"以下特质不可用: {invalid_traits}")
                            st.info(f"可用特质: {available_traits}")
                        else:
                            # 创建结果目录
                            os.makedirs(result_dir, exist_ok=True)
                            
                            # 初始化SJT运行器
                            runner = SJTRunner(
                                situation_theme=situation_theme,
                                scale=neopir,
                                meta=neopir_meta
                            )
                            
                            # 生成题目
                            all_items = runner.cook_async(
                                selected_traits,
                                n_item=n_item,
                                model=model,
                                save_results=True,
                                results_dir=result_dir,
                                detailed_fname=result_detailed_sjt_fn,
                                fname=result_sjt_fn,
                            )
                            
                            st.success("✅ SJT题目生成成功！")
                            
                            # 显示结果摘要
                            st.markdown("### 📊 生成结果摘要")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("生成题目数", len(all_items))
                            with col2:
                                st.metric("涉及特质", len(selected_traits))
                            with col3:
                                st.metric("使用模型", model)
                            with col4:
                                st.metric("情境主题", situation_theme[:10] + "...")
                            
                            # 显示文件保存信息
                            st.info(f"📁 结果已保存到: {result_dir}/{result_sjt_fn}.json")
                            st.info(f"📄 详细结果已保存到: {result_dir}/{result_detailed_sjt_fn}.json")
                            
                            # 显示部分结果预览
                            if all_items:
                                st.markdown("### 🔍 题目预览")
                                for i, item in enumerate(all_items[:3]):  # 只显示前3道题
                                    with st.expander(f"题目 {i+1}"):
                                        st.json(item)
                                
                                if len(all_items) > 3:
                                    st.info(f"还有 {len(all_items)-3} 道题目，请查看保存的文件获取完整结果。")
                    
                    except Exception as e:
                        st.error(f"生成过程中出现错误: {str(e)}")
                        st.info("请检查配置参数或联系开发者")

else:
    st.error("无法加载NEO-PI-R数据，请检查数据文件是否存在")

# 页脚信息
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 2rem;'>
    <p>💡 SJT情境判断测验生成器 - 基于NEO-PI-R人格模型</p>
    <p>🔬 支持多种语言模型 | 🎯 精准人格特质评估 | 📊 详细结果分析</p>
</div>
""", unsafe_allow_html=True)
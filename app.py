import gradio as gr
import asyncio
import threading
import time
import os
from datetime import datetime
from src import SJTRunner, DataLoader
import json

class SJTGradioApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.runner = None
        self.progress_info = {"current": 0, "total": 0, "status": "ready", "details": ""}
        self.results = None

    def load_scale_data(self, scale_name):
        """加载量表数据"""
        try:
            scale_data = self.data_loader.load(scale_name, 'zh')
            scale_meta = self.data_loader.load_meta(scale_name)
            available_traits = list(scale_data.keys())
            trait_names = [f"{trait}: {scale_data[trait]['facet_name']}" for trait in available_traits]
            return scale_data, scale_meta, available_traits, trait_names
        except Exception as e:
            return None, None, [], [f"错误: {str(e)}"]

    def update_traits_choices(self, scale_name):
        """更新特质选择列表"""
        _, _, available_traits, trait_names = self.load_scale_data(scale_name)
        return gr.CheckboxGroup(choices=trait_names, value=[], label="选择特质")

    def extract_trait_codes(self, selected_traits, scale_name):
        """从选择的特质名称中提取特质代码"""
        scale_data, _, _, _ = self.load_scale_data(scale_name)
        if not scale_data:
            return []

        trait_codes = []
        for trait_display in selected_traits:
            trait_code = trait_display.split(":")[0].strip()
            if trait_code in scale_data:
                trait_codes.append(trait_code)
        return trait_codes

    async def run_sjt_generation(self, scale_name, selected_traits, situation_theme,
                                n_item, model_name, result_dir, result_filename):
        """异步运行SJT生成"""
        try:
            # 重置进度信息
            self.progress_info = {"current": 0, "total": 0, "status": "initializing", "details": "正在初始化..."}

            # 加载数据
            scale_data, scale_meta, _, _ = self.load_scale_data(scale_name)
            if not scale_data:
                self.progress_info["status"] = "error"
                self.progress_info["details"] = "无法加载量表数据"
                return

            # 提取特质代码
            trait_codes = self.extract_trait_codes(selected_traits, scale_name)
            if not trait_codes:
                self.progress_info["status"] = "error"
                self.progress_info["details"] = "请选择至少一个特质"
                return

            # 初始化运行器
            self.runner = SJTRunner(
                situation_theme=situation_theme,
                scale=scale_data,
                meta=scale_meta
            )

            self.progress_info["status"] = "running"
            self.progress_info["details"] = f"开始生成 {len(trait_codes)} 个特质的SJT题目..."
            self.progress_info["total"] = len(trait_codes) * n_item

            # 创建结果目录
            os.makedirs(result_dir, exist_ok=True)

            # 运行生成
            results = await self.runner.cook_async(
                trait_codes,
                n_item=n_item,
                model=model_name,
                save_results=True,
                results_dir=result_dir,
                detailed_fname=f"{result_filename}_detailed",
                fname=result_filename,
                progress_callback=self.update_progress
            )

            self.results = results
            self.progress_info["status"] = "completed"
            self.progress_info["details"] = f"生成完成！共生成 {len(results)} 道题目"

        except Exception as e:
            self.progress_info["status"] = "error"
            self.progress_info["details"] = f"生成过程中出现错误: {str(e)}"

    def update_progress(self, current, total, details=""):
        """更新进度信息"""
        self.progress_info["current"] = current
        self.progress_info["total"] = total
        if details:
            self.progress_info["details"] = details

    def get_progress_info(self):
        """获取当前进度信息"""
        info = self.progress_info.copy()
        if info["total"] > 0:
            percentage = (info["current"] / info["total"]) * 100
            info["percentage"] = percentage
        else:
            info["percentage"] = 0
        return info

    def start_generation(self, scale_name, selected_traits, situation_theme,
                        n_item, model_name, result_dir, result_filename):
        """启动生成过程"""
        if not selected_traits:
            return "请选择至少一个特质！"

        # 在新线程中运行异步任务
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.run_sjt_generation(
                    scale_name, selected_traits, situation_theme,
                    n_item, model_name, result_dir, result_filename
                )
            )
            loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

        return "生成任务已启动，请查看进度面板..."

# 创建应用实例
app = SJTGradioApp()

# 自定义CSS
custom_css = """
.progress-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}

.status-box {
    background: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.error-box {
    background: #f8d7da;
    border-left: 4px solid #dc3545;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.success-box {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
"""

# 创建Gradio界面
with gr.Blocks(css=custom_css, title="SJT Generator - Hugging Face Style", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎯 SJT (Situational Judgment Test) Generator

        一个基于人工智能的情境判断测验生成工具，支持多种心理学量表和特质。

        ### 使用说明
        1. 选择心理学量表和目标特质
        2. 设置情境主题和生成参数
        3. 点击开始生成，实时查看进度
        4. 生成完成后查看和下载结果
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## ⚙️ 参数设置")

            # 量表选择
            scale_dropdown = gr.Dropdown(
                choices=["NEO-PI-R", "Big5"],
                value="NEO-PI-R",
                label="📊 心理学量表",
                info="选择要使用的心理学量表"
            )

            # 特质选择（动态更新）
            traits_checkbox = gr.CheckboxGroup(
                choices=[],
                label="🎯 目标特质",
                info="选择要生成题目的心理特质"
            )

            # 其他参数
            situation_theme = gr.Textbox(
                value="大学校园里的日常生活",
                label="🏫 情境主题",
                info="描述测验情境的主题背景"
            )

            with gr.Row():
                n_item = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="📝 每个特质生成题目数",
                    info="为每个选中的特质生成多少道题目"
                )

                model_dropdown = gr.Dropdown(
                    choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
                    value="gpt-4",
                    label="🤖 AI模型",
                    info="选择用于生成的AI模型"
                )

            with gr.Row():
                result_dir = gr.Textbox(
                    value="output",
                    label="📁 结果保存目录",
                    info="生成结果的保存目录"
                )

                result_filename = gr.Textbox(
                    value="sjt-text",
                    label="📄 结果文件名",
                    info="保存文件的名称（不含扩展名）"
                )

            # 生成按钮
            generate_btn = gr.Button(
                "🚀 开始生成 SJT 题目",
                variant="primary",
                size="lg"
            )

            # 状态消息
            status_msg = gr.Textbox(
                label="📢 状态消息",
                interactive=False,
                lines=2
            )

        with gr.Column(scale=1):
            gr.Markdown("## 📊 实时进度")

            # 进度显示
            progress_text = gr.Markdown(
                "### 🔄 等待开始...",
                elem_classes=["status-box"]
            )

            # 自动刷新进度
            def update_progress_display():
                info = app.get_progress_info()

                if info["status"] == "ready":
                    progress_md = "### 🔄 等待开始..."
                elif info["status"] == "initializing":
                    progress_md = "### ⚡ 正在初始化..."
                elif info["status"] == "running":
                    percentage = info.get("percentage", 0)
                    progress_md = f"""
### 🎯 生成进行中...

**进度:** {info['current']}/{info['total']} ({percentage:.1f}%)

**详情:** {info['details']}
                    """
                elif info["status"] == "completed":
                    progress_md = f"""
### ✅ 生成完成！

**结果:** {info['details']}

请查看结果文件夹获取生成的SJT题目。
                    """
                elif info["status"] == "error":
                    progress_md = f"""
### ❌ 生成失败

**错误信息:** {info['details']}

请检查参数设置后重试。
                    """
                else:
                    progress_md = "### 🔄 状态未知..."

                return progress_md

            # 定期更新进度
            progress_timer = gr.Timer(1)  # 每秒更新一次

    # 事件绑定
    scale_dropdown.change(
        app.update_traits_choices,
        inputs=[scale_dropdown],
        outputs=[traits_checkbox]
    )

    generate_btn.click(
        app.start_generation,
        inputs=[
            scale_dropdown, traits_checkbox, situation_theme,
            n_item, model_dropdown, result_dir, result_filename
        ],
        outputs=[status_msg]
    )

    progress_timer.tick(
        update_progress_display,
        outputs=[progress_text]
    )

    # 初始化特质选择
    demo.load(
        app.update_traits_choices,
        inputs=[scale_dropdown],
        outputs=[traits_checkbox]
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )
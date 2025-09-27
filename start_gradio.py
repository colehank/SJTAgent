#!/usr/bin/env python3
"""
SJT Generator Gradio App Launcher

这个脚本用于启动SJT生成器的Gradio Web界面。
运行此脚本后，可以在浏览器中访问 http://localhost:7860 来使用交互式界面。

使用方法：
    python start_gradio.py
"""

import os
import sys
import subprocess


def check_requirements():
    """检查是否安装了必要的依赖包"""
    try:
        import gradio
        import nest_asyncio
        print("✅ 依赖包检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False


def main():
    """主函数"""
    print("🚀 SJT Generator Gradio App")
    print("=" * 40)

    # 检查依赖
    if not check_requirements():
        return

    # 检查环境变量
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  警告: 未检测到 OPENAI_API_KEY 环境变量")
        print("   请确保已设置 OpenAI API Key")

    # 启动应用
    print("🌐 启动 Gradio 应用...")
    print("📍 应用将在 http://127.0.0.1:7860 启动")
    print("💡 按 Ctrl+C 停止应用")
    print("-" * 40)

    try:
        # 导入并启动Gradio应用
        from gradio_app import demo
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
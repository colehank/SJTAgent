# SJTAgent

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)


基于大语言模型的 **情境判断测验(SJT)** 自动生成系统。

## 项目简介

**SJTAgent** 是一个使用大语言/视觉模型(LLM/LVM)自动生成个人情境判断测验(Personal Situation Judgment Test, PSJT)的智能系统。该项目通过多阶段工作流程，将传统的自评量表题目转换为更加真实有效的情境化评估题目。

### 主要特性

- **多阶段工作流程**：智能化的题目生成流程  
- **内置题库**：[NEO-PI-R / IPIP / Mussel's SJT的中英文题目](src/datasets/scales)，可通过`src.data_loader`模块加载
- **内置方法**：[Li, Krumm的SJT生成方法](make_baseline_sjt.py), 已生成在`results/SJTs`中
- **质量评估**：[内置题目质量评估功能](eval_aigs.py)


## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/colehank/SJTAgent.git
cd SJTAgent
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 编辑 .env 文件，添加你的 API 配置
# OPENAI_API_KEY=your_api_key_here
# OPENAI_BASE_URL=your_base_url_here
```

### 3. 使用示例

#### 3.1 生成 SJT 题目

见 `SJTAgent-text.py` 
- **输出格式**：JSON 文件 + Word 文档

### 4. 题目质量评估

使用内置的成对比较评估功能：

```bash
# 详细使用方法请参考
python eval_aigs.py
```

## 贡献指南

我们热烈欢迎社区贡献！如果你有任何想法或发现了问题：

1. **报告 Bug**：请在 [Issues](https://github.com/colehank/SJTAgent/issues) 中详细描述问题
2. **功能建议**：欢迎提出新的功能想法
3. **代码贡献**：欢迎提交 Pull Request

### 贡献步骤

```bash
# 1. Fork 项目
# 2. 创建分支
git checkout -b feature/your-feature-name

# 3. 提交更改
git commit -m "Add your feature description"

# 4. 推送到分支
git push origin feature/your-feature-name

# 5. 创建 Pull Request
```
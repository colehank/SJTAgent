import asyncio
import json
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional, Annotated
from dataclasses import dataclass, asdict
from datetime import datetime

import os
from tqdm import tqdm
import operator
import nest_asyncio
import tiktoken

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

@dataclass
class PairwiseEvaluation:
    """配对评估结果数据结构"""
    item1_id: str
    item2_id: str
    dimension: str
    winner: str
    evaluation_time: str

@dataclass
class TokenUsage:
    """Token使用统计"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, input_tokens: int, output_tokens: int):
        """添加token使用量"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += (input_tokens + output_tokens)

@dataclass 
class CostConfig:
    """费用配置"""
    input_token_rate: float  # 输入token费率 (per 1M tokens)
    output_token_rate: float  # 输出token费率 (per 1M tokens)
    
    def calculate_cost(self, token_usage: TokenUsage) -> float:
        """计算总费用"""
        input_cost = (token_usage.input_tokens / 1_000_000) * self.input_token_rate
        output_cost = (token_usage.output_tokens / 1_000_000) * self.output_token_rate
        return input_cost + output_cost

class DimensionEvaluation(BaseModel):
    """单个维度评估结果的Pydantic模型，用于结构化输出"""
    class Config:
        extra = "forbid"  # 禁止额外字段
    
    def __init__(self, **data):
        # 动态添加维度字段
        for key, value in data.items():
            if key not in self.__fields__:
                self.__fields__[key] = Field(..., description=f"Evaluation result for {key} dimension")
        super().__init__(**data)

def create_dimension_model(dimensions: list[dict[str, str]]) -> type:
    """动态创建包含所有维度的Pydantic模型"""
    from pydantic import create_model
    
    # 简化字段定义，去掉复杂的配置
    fields = {}
    for dim in dimensions:
        field_name = dim['name']
        fields[field_name] = (str, Field(..., description=f"Choose 'A' or 'B' for {dim['description']}"))
    
    # 使用 pydantic.create_model 动态创建模型
    DynamicDimensionModel = create_model('DynamicDimensionModel', **fields)
    
    return DynamicDimensionModel

class EvaluationState(TypedDict):
    """LangGraph状态定义 - 使用TypedDict和reducers"""
    test_items: dict[str, dict[str, Any]]
    dimensions: list[dict[str, str]]
    
    # 处理过程数据 - 使用reducers避免并发更新冲突
    pairs_to_evaluate: Annotated[list[tuple[str, str, str]], operator.add]
    completed_evaluations: Annotated[list[PairwiseEvaluation], operator.add]
    current_batch: list[tuple[str, str, str]]
    batch_results: Annotated[list[PairwiseEvaluation], operator.add]
    
    # 输出数据
    final_results: Optional[pd.DataFrame]
    
    # 配置
    batch_size: int
    max_concurrent: int
    cost_config: Optional[CostConfig]
    
    # Token统计
    token_usage: TokenUsage
    
    # 进度条相关
    total_pairs: int
    progress_bar: Optional[tqdm]
    show_progress: bool


class PsychologicalItemEvaluator:
    """心理测验题目评估器"""
    
    def __init__(
        self, 
        model_name: str = "qwen-plus", 
        temperature: float = 0.3, 
        api_key: Optional[str] = None, 
        cost_config: Optional[CostConfig] = None,
        sys_prompt: str = "你是一位心理测验专家，负责评估题目质量。"
    ) -> None:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            raise ValueError("请设置OPENAI_API_KEY环境变量或在初始化时传入api_key参数")
        
        self.sys_prompt = sys_prompt
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        self.cost_config = cost_config or CostConfig(
            input_token_rate=0.8,
            output_token_rate=2.0
        )
        
        # 初始化JSON解析器（将在评估时动态创建）
        self.json_parser = None
        self.dimension_model = None
        
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        return len(self.tokenizer.encode(text))
    
    def setup_structured_output(self, dimensions: list[dict[str, str]]) -> None:
        """设置结构化输出解析器"""
        # 创建动态维度模型
        self.dimension_model = create_dimension_model(dimensions)
        # 创建JSON解析器
        self.json_parser = JsonOutputParser(pydantic_object=self.dimension_model)
    
    def estimate_cost_for_evaluation(
        self,
        test_items: dict[str, dict[str, Any]],
        dimensions: list[dict[str, str]]
    ) -> tuple[TokenUsage, float]:
        """预估评估的token使用量和费用"""
        
        # 计算配对数
        total_pairs = len(list(combinations(test_items.keys(), 2)))
        
        # 创建样本提示词来估算token使用
        sample_items = list(test_items.items())[:2] if len(test_items) >= 2 else list(test_items.items())
        
        if len(sample_items) < 2:
            # 如果题目不足2个，复制第一个题目作为样本
            sample_items = [sample_items[0], sample_items[0]]
        
        # print(sample_items)
        sample_prompt = self.create_single_eval(
            sample_items[0][1], sample_items[1][1], 
            dimensions
        )
        
        # 计算单次调用的token使用
        input_tokens_per_call = self.count_tokens(self.sys_prompt) + self.count_tokens(sample_prompt)
        
        # 估算输出token（基于维度数量和经验值）
        estimated_output_per_dimension = 10  # 每个维度约输出10个token
        output_tokens_per_call = len(dimensions) * estimated_output_per_dimension + 20  # 额外20个token用于JSON格式
        
        # 计算总的token使用量
        total_input_tokens = input_tokens_per_call * total_pairs
        total_output_tokens = output_tokens_per_call * total_pairs
        
        estimated_usage = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens
        )
        
        estimated_cost = self.cost_config.calculate_cost(estimated_usage)
        
        return estimated_usage, estimated_cost
    
    def create_evaluation_workflow(self) -> StateGraph:
        """创建LangGraph工作流"""
        workflow = StateGraph(EvaluationState)  # 使用定义的EvaluationState类
        
        workflow.add_node("generate_pairs", self.generate_pairs)
        workflow.add_node("batch_evaluations", self.batch_evaluations)
        workflow.add_node("process_batch", self.process_batch)
        workflow.add_node("aggregate_results", self.aggregate_results)
        workflow.add_node("create_dataframe", self.create_dataframe)
        
        workflow.set_entry_point("generate_pairs")
        workflow.add_edge("generate_pairs", "batch_evaluations")
        workflow.add_conditional_edges(
            "batch_evaluations",
            self.should_continue_batching,
            {
                "continue": "process_batch",
                "end": "aggregate_results"
            }
        )
        workflow.add_edge("process_batch", "batch_evaluations")
        workflow.add_edge("aggregate_results", "create_dataframe")
        workflow.add_edge("create_dataframe", END)
        
        return workflow.compile()
    
    def generate_pairs(self, state: EvaluationState) -> dict[str, Any]:
        """生成所有需要评估的配对（每个配对评估所有维度）"""
        pairs = []
        item_ids = list(state['test_items'].keys())
        
        # 只生成不同方法间的题目配对，每个配对将评估所有维度
        for item1, item2 in combinations(item_ids, 2):
            # 提取方法类型（假设ID格式为"method_index"）
            item1_type = item1.split('_')[0]
            item2_type = item2.split('_')[0]
            
            # 只添加不同方法间的配对
            if item1_type != item2_type:
                pairs.append((item1, item2, None))  # None表示评估所有维度
        
        # 初始化进度条
        progress_bar = None
        if state.get('show_progress', True):
            progress_bar = tqdm(
                total=len(pairs), 
                desc="🔍 评估进度", 
                unit="pair",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        
        return {
            'pairs_to_evaluate': pairs,
            'completed_evaluations': [],
            'total_pairs': len(pairs),
            'progress_bar': progress_bar
        }
    
    def batch_evaluations(self, state: EvaluationState) -> dict[str, Any]:
        """准备下一批评估"""
        remaining_pairs = [
            pair for pair in state['pairs_to_evaluate'] 
            if not any(
                eval.item1_id == pair[0] and eval.item2_id == pair[1]
                for eval in state['completed_evaluations']
            )
        ]
        
        # 取下一批
        current_batch = remaining_pairs[:state['batch_size']]
        
        return {
            'current_batch': current_batch,
            'batch_results': []
        }
    
    def process_batch(self, state: EvaluationState) -> dict[str, Any]:
        """处理当前批次的评估（同步版本）"""
        
        async def async_batch_processing():
            semaphore = asyncio.Semaphore(state['max_concurrent'])
            
            async def evaluate_pair_with_progress(item1_id: str, item2_id: str):
                async with semaphore:
                    results, token_usage = await self.evaluate_pair_all_dimensions_async(
                        state['test_items'][item1_id],
                        state['test_items'][item2_id],
                        item1_id,
                        item2_id,
                        state['dimensions']
                    )
                    
                    # 更新token使用统计
                    state['token_usage'].add(token_usage.input_tokens, token_usage.output_tokens)
                    
                    # 更新进度条
                    if state.get('progress_bar'):
                        state['progress_bar'].update(1)
                        # 显示当前评估的详细信息
                        current_cost = state['cost_config'].calculate_cost(state['token_usage']) if state.get('cost_config') else 0
                        state['progress_bar'].set_postfix({
                            '题目': f"{item1_id}-{item2_id}",
                            '维度数': len(results),
                            '费用': f"${current_cost:.4f}"
                        })
                    
                    return results
            
            # 并行处理当前批次
            tasks = [
                evaluate_pair_with_progress(item1_id, item2_id)
                for item1_id, item2_id, _ in state['current_batch']
            ]
            
            batch_results = await asyncio.gather(*tasks)
            # 展平结果列表，因为每个任务现在返回多个评估结果
            flattened_results = []
            for results in batch_results:
                flattened_results.extend(results)
            
            return flattened_results
        
        # 检查是否在Jupyter环境中并处理事件循环
        try:
            loop = asyncio.get_running_loop()
            nest_asyncio.apply()
            batch_results = asyncio.run(async_batch_processing())
        except RuntimeError:
            batch_results = asyncio.run(async_batch_processing())
        
        return {
            'batch_results': batch_results,
            'completed_evaluations': batch_results
        }
    
    async def evaluate_pair_all_dimensions_async(
        self,
        item1: dict[str, Any],
        item2: dict[str, Any],
        item1_id: str,
        item2_id: str,
        dimensions: list[dict[str, str]]
    ) -> tuple[list[PairwiseEvaluation], TokenUsage]:
        """异步评估单个配对的所有维度，使用结构化输出"""
        
        # 确保结构化输出解析器已设置
        if self.json_parser is None or self.dimension_model is None:
            self.setup_structured_output(dimensions)
        
        prompt = self.create_single_eval(
            item1, item2, 
            dimensions
        )
        
        # 创建带有结构化输出指令的LLM链
        format_instructions = self.json_parser.get_format_instructions()
        
        system_content = f"{self.sys_prompt}\n\n{format_instructions}"
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=prompt)
        ]
        
        # 计算输入token数量
        input_tokens = self.count_tokens(system_content) + self.count_tokens(prompt)
        
        try:
            # 使用结构化输出链
            chain = self.llm | self.json_parser
            response_dict = await chain.ainvoke(messages)
            response_content = json.dumps(response_dict, ensure_ascii=False)
            
        except Exception as e:
            print(f"⚠️  结构化输出解析失败，使用回退方式: {str(e)[:100]}")
            # 回退到原始方法
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            response_dict = self._parse_multi_dimension_evaluation_response_fallback(
                response_content, dimensions
            )
        
        # 计算输出token数量
        output_tokens = self.count_tokens(response_content)
        
        # 创建token使用统计
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        # 创建多个PairwiseEvaluation对象
        evaluations = []
        
        # 检查是否有有效的评估结果
        if not response_dict:
            print(f"⚠️  配对 {item1_id}-{item2_id} 的所有维度评估均失败，跳过此配对")
            return [], token_usage
        
        for dimension_name, winner in response_dict.items():
            # 只为成功解析的维度创建评估对象
            evaluations.append(PairwiseEvaluation(
                item1_id=item1_id,
                item2_id=item2_id,
                dimension=dimension_name,
                winner=winner,
                evaluation_time=datetime.now().isoformat()
            ))
        
        # 如果只有部分维度成功，给出警告
        if len(evaluations) < len(dimensions):
            failed_dims = len(dimensions) - len(evaluations)
            print(f"⚠️  配对 {item1_id}-{item2_id} 有 {failed_dims} 个维度评估失败")
        
        return evaluations, token_usage
    
    def create_single_eval(
        self,
        item1: dict[str, Any],
        item2: dict[str, Any],
        dimensions: list[dict[str, str]]
    ) -> str:
        """创建多维度评估提示词，优化结构化输出"""
        # 构建维度说明
        dimension_descriptions = []
        for i, dim in enumerate(dimensions, 1):
            dimension_descriptions.append(f"{i}. {dim['name']}: {dim['description']}")
        
        # 构建期望的输出格式说明（简化版本）
        output_example = {}
        for dim in dimensions:
            output_example[dim['name']] = "A"  # 示例值
        
        prompt = f"""请比较以下两个情景判断测验题目在多个维度上的质量。

评估维度：
{chr(10).join(dimension_descriptions)}

题目A：
情境：{item1['situation']}
选项：{json.dumps(item1['options'], ensure_ascii=False, indent=2)}

题目B：
情境：{item2['situation']}
选项：{json.dumps(item2['options'], ensure_ascii=False, indent=2)}

请对每个维度分别评估哪个题目更好，选择"A"或"B"。

输出示例格式：
{json.dumps(output_example, ensure_ascii=False, indent=2)}

注意：只能选择"A"或"B"，不允许其他值。"""
        
        return prompt

    def _parse_multi_dimension_evaluation_response_fallback(
        self, 
        response: str, 
        dimensions: list[dict[str, str]]
    ) -> dict[str, str]:
        """回退方案：解析多维度LLM评估响应（使用原始JSON解析方法）"""
        try:
            parsed = self._parse_json_with_retry(response)
            
            # 验证并清理结果
            results = {}
            for dim in dimensions:
                dim_name = dim['name']
                if dim_name in parsed:
                    winner = str(parsed[dim_name]).upper()
                    if winner in ['A', 'B']:
                        results[dim_name] = winner
                    else:
                        # 无法解析时跳过该维度，而不是设置默认值
                        print(f"⚠️  维度 {dim_name} 解析结果无效: {parsed[dim_name]}")
                        continue
                else:
                    print(f"⚠️  维度 {dim_name} 在响应中缺失")
                    continue
            
            return results
            
        except (json.JSONDecodeError, KeyError) as e:
            # 如果JSON解析失败，返回空字典
            print(f"❌ JSON解析完全失败: {str(e)}")
            return {}

    def _parse_json_with_retry(self, response: str, max_retries: int = 20) -> dict:
        """带重试机制的JSON解析"""
        import re
        
        for attempt in range(max_retries):
            try:
                # 尝试直接解析JSON
                if response.strip().startswith('{'):
                    return json.loads(response)
                else:
                    # 如果不是纯JSON，尝试提取JSON部分
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        raise json.JSONDecodeError("No JSON found", "", 0)
                        
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    # 记录重试
                    print(f"⚠️  JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    # 可以在这里添加重新调用LLM的逻辑
                    continue
                else:
                    print(f"❌ JSON解析最终失败，已达到最大重试次数 ({max_retries})")
                    raise
        
        return {}
    
    def _parse_multi_dimension_evaluation_response(
        self, 
        response: str, 
        dimensions: list[dict[str, str]]
    ) -> dict[str, str]:
        """解析多维度LLM评估响应"""
        try:
            parsed = self._parse_json_with_retry(response)
            
            # 验证并清理结果
            results = {}
            for dim in dimensions:
                dim_name = dim['name']
                if dim_name in parsed:
                    winner = str(parsed[dim_name]).upper()
                    if winner in ['A', 'B']:
                        results[dim_name] = winner
                    else:
                        results[dim_name] = 'None'
                else:
                    results[dim_name] = 'None'
            
            return results
            
        except (json.JSONDecodeError, KeyError):
            # 如果JSON解析失败，为所有维度返回默认值
            return {dim['name']: 'None' for dim in dimensions}
    
    def should_continue_batching(self, state: EvaluationState) -> str:
        """判断是否继续批处理"""
        remaining_pairs = [
            pair for pair in state['pairs_to_evaluate'] 
            if not any(
                eval.item1_id == pair[0] and eval.item2_id == pair[1]
                for eval in state['completed_evaluations']
            )
        ]
        
        return "continue" if remaining_pairs else "end"
    
    def aggregate_results(self, state: EvaluationState) -> dict[str, Any]:
        """聚合评估结果"""
        # 完成进度条
        if state.get('progress_bar'):
            final_cost = state['cost_config'].calculate_cost(state['token_usage']) if state.get('cost_config') else 0
            state['progress_bar'].set_postfix({
                '状态': '完成',
                '总计': f"{len(state['completed_evaluations'])}个",
                '最终费用': f"${final_cost:.4f}"
            })
            state['progress_bar'].close()
        
        # 显示完成信息
        total_evaluations = len(state['completed_evaluations'])
        total_dimensions = len({eval.dimension for eval in state['completed_evaluations']})
        total_items = len({
            eval.item1_id for eval in state['completed_evaluations']
        }.union({
            eval.item2_id for eval in state['completed_evaluations']
        }))
        
        print(f"\n🎉 评估完成!")
        print(f"📊 总计: {total_evaluations} 个配对评估")
        print(f"📏 维度: {total_dimensions} 个评估维度")
        print(f"📝 题目: {total_items} 个不同题目")
        
        # 显示token使用和费用统计
        if state.get('token_usage'):
            token_usage = state['token_usage']
            print(f"\n💰 费用统计:")
            print(f"   输入tokens: {token_usage.input_tokens:,}")
            print(f"   输出tokens: {token_usage.output_tokens:,}")
            print(f"   总tokens: {token_usage.total_tokens:,}")
            
            if state.get('cost_config'):
                final_cost = state['cost_config'].calculate_cost(token_usage)
                input_cost = (token_usage.input_tokens / 1_000_000) * state['cost_config'].input_token_rate
                output_cost = (token_usage.output_tokens / 1_000_000) * state['cost_config'].output_token_rate
                print(f"   输入费用: ${input_cost:.4f}")
                print(f"   输出费用: ${output_cost:.4f}")
                print(f"   总费用: ${final_cost:.4f}")
        
        return {}
    
    def create_dataframe(self, state: EvaluationState) -> dict[str, Any]:
        """创建最终的DataFrame结果"""
        data = [asdict(eval) for eval in state['completed_evaluations']]
        df = pd.DataFrame(data)
        
        # 显示结果摘要
        if len(df) > 0:
            # print(f"\n📋 结果摘要:")
            # print(f"   平均置信度: {df['confidence'].mean():.3f}")
            # print(f"   置信度范围: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
            
            # 按维度显示胜率统计
            print(f"\n🏆 各维度胜率统计:")
            for dimension in df['dimension'].unique():
                dim_data = df[df['dimension'] == dimension]
                print(f"   {dimension}:")
                
                # 计算每个题目的胜率
                win_counts = {}
                total_counts = {}
                
                for _, row in dim_data.iterrows():
                    winner_id = row['item1_id'] if row['winner'] == 'A' else row['item2_id']
                    loser_id = row['item2_id'] if row['winner'] == 'A' else row['item1_id']
                    
                    win_counts[winner_id] = win_counts.get(winner_id, 0) + 1
                    win_counts[loser_id] = win_counts.get(loser_id, 0)
                    
                    total_counts[row['item1_id']] = total_counts.get(row['item1_id'], 0) + 1
                    total_counts[row['item2_id']] = total_counts.get(row['item2_id'], 0) + 1
                
                # 计算并显示胜率
                win_rates = {
                    item_id: win_counts.get(item_id, 0) / total_counts.get(item_id, 1)
                    for item_id in total_counts
                }
                
                sorted_items = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
                for i, (item_id, rate) in enumerate(sorted_items[:3], 1):
                    print(f"     {i}. 题目{item_id}: {rate:.1%} 胜率")
        df.rename(columns={'item1_id': 'A', 'item2_id': 'B'}, inplace=True)
        
        return {'final_results': df}
            
    def evaluate_test_items(
        self,
        test_items: dict[str, dict[str, Any]],
        dimensions: list[dict[str, str]],
        batch_size: int = 10,
        max_concurrent: int = 5,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """主要的评估入口函数（同步版本）"""
        
        # 预估费用
        estimated_usage, estimated_cost = self.estimate_cost_for_evaluation(test_items, dimensions)
        
        # 显示开始信息
        total_items = len(test_items)
        total_dimensions = len(dimensions)
        total_pairs = len(list(combinations(test_items.keys(), 2)))  # 只计算配对数，不乘以维度数
        
        if show_progress:
            print(f"🚀 开始心理测验题目评估")
            print(f"📝 题目数量: {total_items}")
            print(f"📏 评估维度: {total_dimensions}")
            print(f"🔍 配对总数: {total_pairs}（每对评估{total_dimensions}个维度）")
            print(f"⚙️  批处理大小: {batch_size}")
            print(f"🔄 最大并发: {max_concurrent}")
            print(f"💡 优化: 单次API调用评估所有维度，减少{(total_dimensions-1)*total_pairs}次调用")
            
            # 显示预估费用
            print(f"\n💰 预估费用:")
            print(f"   预估输入tokens: {estimated_usage.input_tokens:,}")
            print(f"   预估输出tokens: {estimated_usage.output_tokens:,}")
            print(f"   预估总tokens: {estimated_usage.total_tokens:,}")
            print(f"   预估总费用: ${estimated_cost:.4f}")
            print(f"   费率配置: 输入${self.cost_config.input_token_rate}/1M tokens, 输出${self.cost_config.output_token_rate}/1M tokens")
            
            print("-" * 50)
            
            # # 询问用户是否继续
            # user_input = input("是否继续执行评估？(y/n): ").lower().strip()
            # if user_input not in ['y', 'yes', '是']:
            #     print("评估已取消")
            #     return pd.DataFrame()
        
        # 创建初始状态（使用正确的类型）
        initial_state = {
            'test_items': test_items,
            'dimensions': dimensions,
            'pairs_to_evaluate': [],
            'completed_evaluations': [],
            'current_batch': [],
            'batch_results': [],
            'final_results': None,
            'batch_size': batch_size,
            'max_concurrent': max_concurrent,
            'show_progress': show_progress,
            'total_pairs': 0,
            'progress_bar': None,
            'cost_config': self.cost_config,
            'token_usage': TokenUsage()
        }
        
        try:
            # 创建并运行工作流
            workflow = self.create_evaluation_workflow()
            final_state = workflow.invoke(initial_state)
            
            # 返回最终结果
            return final_state.get('final_results')
            
        except KeyboardInterrupt:
            # 处理用户中断
            if initial_state.get('progress_bar'):
                initial_state['progress_bar'].close()
            print("\n⚠️  评估被用户中断")
            return pd.DataFrame()
        except Exception as e:
            # 处理其他异常
            if initial_state.get('progress_bar'):
                initial_state['progress_bar'].close()
            print(f"\n❌ 评估过程中出现错误: {e}")
            raise
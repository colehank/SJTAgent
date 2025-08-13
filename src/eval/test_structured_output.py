#!/usr/bin/env python3
"""
测试结构化输出功能的简单脚本
"""

import json
from item_eval import (
    create_dimension_model, 
    PsychologicalItemEvaluator,
    CostConfig
)
from langchain_core.output_parsers import JsonOutputParser

def test_dimension_model_creation():
    """测试动态维度模型创建"""
    print("🧪 测试动态维度模型创建...")
    
    # 定义测试维度
    dimensions = [
        {
            "name": "TestDimension1",
            "description": "这是测试维度1"
        },
        {
            "name": "TestDimension2", 
            "description": "这是测试维度2"
        }
    ]
    
    try:
        # 创建动态模型
        DynamicModel = create_dimension_model(dimensions)
        
        # 创建解析器
        parser = JsonOutputParser(pydantic_object=DynamicModel)
        
        # 获取格式指令
        format_instructions = parser.get_format_instructions()
        print("✅ 成功创建动态模型")
        print("📋 格式指令:")
        print(format_instructions)
        
        # 测试有效数据
        valid_data = {"TestDimension1": "A", "TestDimension2": "B"}
        instance = DynamicModel(**valid_data)
        print(f"✅ 有效数据测试通过: {instance.dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 动态模型创建失败: {e}")
        return False

def test_structured_output_parsing():
    """测试结构化输出解析"""
    print("\n🧪 测试结构化输出解析...")
    
    dimensions = [
        {
            "name": "Quality",
            "description": "题目质量评估"
        },
        {
            "name": "Clarity", 
            "description": "题目清晰度评估"
        }
    ]
    
    try:
        # 创建评估器实例（不需要真实的API密钥来测试模型创建）
        import os
        os.environ["OPENAI_API_KEY"] = "test-key-for-model-creation"
        
        evaluator = PsychologicalItemEvaluator(
            cost_config=CostConfig(input_token_rate=0.0, output_token_rate=0.0)
        )
        
        # 设置结构化输出
        evaluator.setup_structured_output(dimensions)
        
        print("✅ 结构化输出解析器设置成功")
        print(f"📋 解析器类型: {type(evaluator.json_parser)}")
        print(f"📋 模型字段: {list(evaluator.dimension_model.__fields__.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 结构化输出解析器设置失败: {e}")
        return False

def test_fallback_parsing():
    """测试回退解析功能"""
    print("\n🧪 测试回退解析功能...")
    
    dimensions = [
        {"name": "TestDim1", "description": "测试维度1"},
        {"name": "TestDim2", "description": "测试维度2"}
    ]
    
    try:
        import os
        os.environ["OPENAI_API_KEY"] = "test-key"
        
        evaluator = PsychologicalItemEvaluator()
        
        # 测试有效JSON
        valid_json = '{"TestDim1": "A", "TestDim2": "B"}'
        result = evaluator._parse_multi_dimension_evaluation_response_fallback(
            valid_json, dimensions
        )
        print(f"✅ 有效JSON解析: {result}")
        
        # 测试无效JSON
        invalid_json = '{"TestDim1": "A", "TestDim2": "C"}'  # C是无效值
        result = evaluator._parse_multi_dimension_evaluation_response_fallback(
            invalid_json, dimensions
        )
        print(f"✅ 部分无效JSON解析: {result}")
        
        # 测试完全无效JSON
        broken_json = 'not a json at all'
        result = evaluator._parse_multi_dimension_evaluation_response_fallback(
            broken_json, dimensions
        )
        print(f"✅ 无效JSON解析: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 回退解析测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🚀 开始结构化输出功能测试\n")
    
    tests = [
        test_dimension_model_creation,
        test_structured_output_parsing,
        test_fallback_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！结构化输出功能正常工作")
    else:
        print("⚠️ 部分测试失败，请检查实现")
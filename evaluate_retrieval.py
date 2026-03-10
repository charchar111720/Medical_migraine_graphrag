#!/usr/bin/env python3
"""
GraphRAG检索结果评估工具

评估维度：
1. 准确性（Accuracy）：答案是否正确
2. 相关性（Relevance）：检索内容是否相关
3. 完整性（Completeness）：是否包含所有必要信息
4. 一致性（Consistency）：多次查询结果是否一致
5. 响应时间（Latency）：查询速度
6. 可解释性（Explainability）：是否有证据支持
"""

import subprocess
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GraphRAGRetrievalEvaluator:
    """GraphRAG检索结果评估器"""
    
    def __init__(self, root_dir: str = ".", use_poetry: bool = True):
        """
        Args:
            root_dir: GraphRAG项目根目录
            use_poetry: 是否使用poetry run poe方式执行命令
        """
        self.root_dir = Path(root_dir).resolve()
        self.use_poetry = use_poetry
        self.results = []
        logger.info(f"GraphRAG根目录: {self.root_dir}")
        logger.info(f"使用Poetry执行命令: {self.use_poetry}")
        
        # 检查必要的文件/目录
        self._check_environment()
    
    def _check_environment(self):
        """检查运行环境"""
        # 检查GraphRAG配置文件
        config_file = self.root_dir / "graphrag.yaml"
        if not config_file.exists():
            logger.warning(f"未找到GraphRAG配置文件: {config_file}")
        
        # 检查数据目录
        data_dir = self.root_dir / "data"
        if not data_dir.exists():
            logger.warning(f"未找到数据目录: {data_dir}")
        
        # 检查poetry是否可用
        if self.use_poetry:
            try:
                result = subprocess.run(
                    ["poetry", "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Poetry版本: {result.stdout.strip()}")
                else:
                    logger.error("Poetry不可用，请检查安装")
            except FileNotFoundError:
                logger.error("未找到poetry命令，请安装poetry")
    
    def query_graphrag(
        self, 
        query: str, 
        method: str = "local"
    ) -> Tuple[str, float, Dict]:
        """
        执行GraphRAG查询
        
        Args:
            query: 查询问题
            method: 查询方法（local/global/drift）
        
        Returns:
            (答案, 耗时, 元数据)
        """
        start_time = time.time()
        metadata = {"error": True, "method": method}
        
        try:
            # 构建命令 - 使用poetry run poe方式
            if self.use_poetry:
                cmd = [
                    "poetry", "run", "poe", "query",
                    "--root", "./",
                    "--method", method,
                    "--query", query
                ]
            else:
                cmd = [
                    "python", "-m", "graphrag.query",
                    "--root", str(self.root_dir),
                    "--method", method,
                    "--query", query
                ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行查询
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.root_dir)
            )
            
            elapsed = time.time() - start_time
            
            # 记录命令输出
            logger.info(f"命令返回码: {result.returncode}")
            logger.info(f"标准输出: {result.stdout[:500]}...")
            logger.info(f"标准错误: {result.stderr[:500]}...")
            
            if result.returncode == 0:
                answer = result.stdout.strip()
                metadata["error"] = False
                
                # 提取元数据
                metadata.update(self._extract_metadata(answer, method))
                
                logger.info(f"查询成功，耗时: {elapsed:.2f}s，答案长度: {len(answer)}")
                return answer, elapsed, metadata
            else:
                error_msg = f"命令执行失败: {result.stderr}"
                logger.error(error_msg)
                return error_msg, elapsed, metadata
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            error_msg = f"查询超时（{elapsed:.2f}s）"
            logger.error(error_msg)
            metadata["timeout"] = True
            return error_msg, elapsed, metadata
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"执行异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, elapsed, metadata
    
    def _extract_metadata(self, answer: str, method: str) -> Dict:
        """从答案中提取元数据"""
        metadata = {
            "has_sources": False,
            "has_evidence": False,
            "num_entities": 0,
            "num_sources": 0,
            "answer_length": len(answer)
        }
        
        if not answer or "ERROR" in answer:
            return metadata
        
        # 检查是否有来源引用
        if any(keyword in answer for keyword in ["来源", "Source", "根据", "参考"]):
            metadata["has_sources"] = True
            
            # 统计来源数量
            source_pattern = r'(?:来源|Source|根据|参考)[：:]\s*([^。\n]+)'
            sources = re.findall(source_pattern, answer)
            metadata["num_sources"] = len(sources)
        
        # 检查是否有证据等级
        if any(keyword in answer for keyword in ["证据等级", "Evidence", "推荐等级"]):
            metadata["has_evidence"] = True
        
        # 统计实体提及
        entity_pattern = r'(利扎曲普坦|托吡酯|偏头痛|癫痫|曲普坦类|NSAIDs|对乙酰氨基酚|瑞美吉泮|丙戊酸钠)'
        entities = re.findall(entity_pattern, answer)
        metadata["num_entities"] = len(set(entities))
        
        return metadata
    
    def evaluate_accuracy(
        self, 
        query: str, 
        answer: str, 
        ground_truth: str
    ) -> Dict:
        """
        评估准确性
        """
        metrics = {
            'keyword_recall': 0.0,
            'keyword_precision': 0.0,
            'keyword_f1': 0.0,
            'number_accuracy': 0.0,
            'exact_match': 0.0,
            'contains_answer': 0.0
        }
        
        # 如果答案包含错误，直接返回0
        if "ERROR" in answer or "执行失败" in answer or "超时" in answer:
            return metrics
        
        # 1. 关键词匹配（改进：使用更宽松的匹配）
        gt_keywords = set(re.findall(r'[\u4e00-\u9fff]+', ground_truth))
        answer_keywords = set(re.findall(r'[\u4e00-\u9fff]+', answer))
        
        if gt_keywords and answer_keywords:
            intersection = gt_keywords & answer_keywords
            keyword_recall = len(intersection) / len(gt_keywords)
            keyword_precision = len(intersection) / len(answer_keywords)
            keyword_f1 = 2 * (keyword_recall * keyword_precision) / (keyword_recall + keyword_precision) if (keyword_recall + keyword_precision) > 0 else 0
            
            metrics['keyword_recall'] = keyword_recall
            metrics['keyword_precision'] = keyword_precision
            metrics['keyword_f1'] = keyword_f1
        
        # 2. 数值一致性
        gt_numbers = set(re.findall(r'\d+(?:\.\d+)?', ground_truth))
        answer_numbers = set(re.findall(r'\d+(?:\.\d+)?', answer))
        
        if gt_numbers:
            number_match = len(gt_numbers & answer_numbers) / len(gt_numbers)
            metrics['number_accuracy'] = number_match
        
        # 3. 完全匹配（严格）
        metrics['exact_match'] = 1.0 if ground_truth.strip() in answer else 0.0
        
        # 4. 包含关系（宽松）
        gt_tokens = ground_truth.split()
        match_count = sum(1 for token in gt_tokens if len(token) > 1 and token in answer)
        if gt_tokens:
            metrics['contains_answer'] = min(1.0, match_count / len(gt_tokens))
        
        return metrics
    
    def evaluate_relevance(
        self, 
        query: str, 
        answer: str,
        method: str
    ) -> Dict:
        """
        评估相关性
        """
        metrics = {
            'query_coverage': 0.0,
            'length_score': 0.0,
            'has_error': True,
            'error_score': 0.0
        }
        
        # 如果答案包含错误
        if "ERROR" in answer or "执行失败" in answer or "超时" in answer:
            return metrics
        
        metrics['has_error'] = False
        metrics['error_score'] = 1.0
        
        # 1. 查询词覆盖（改进）
        query_keywords = set(re.findall(r'[\u4e00-\u9fff]+', query))
        answer_keywords = set(re.findall(r'[\u4e00-\u9fff]+', answer))
        
        if query_keywords and answer_keywords:
            query_coverage = len(query_keywords & answer_keywords) / len(query_keywords)
            metrics['query_coverage'] = query_coverage
        
        # 2. 答案长度合理性
        answer_length = len(answer)
        
        # 根据查询方法判断合理长度
        if method == "local":
            if 100 <= answer_length <= 500:
                metrics['length_score'] = 1.0
            elif 50 <= answer_length < 100 or 500 < answer_length <= 1000:
                metrics['length_score'] = 0.7
            elif answer_length > 0:
                metrics['length_score'] = 0.5
        elif method == "global":
            if 300 <= answer_length <= 1500:
                metrics['length_score'] = 1.0
            elif 150 <= answer_length < 300 or 1500 < answer_length <= 2500:
                metrics['length_score'] = 0.7
            elif answer_length > 0:
                metrics['length_score'] = 0.5
        elif method == "drift":
            metrics['length_score'] = 1.0 if answer_length > 200 else 0.5
        
        return metrics
    
    def evaluate_completeness(
        self,
        query: str,
        answer: str,
        required_fields: List[str]
    ) -> Dict:
        """
        评估完整性
        """
        metrics = {
            'field_coverage': 0.0,
            'missing_fields': required_fields.copy()
        }
        
        # 如果答案包含错误，直接返回
        if "ERROR" in answer or "执行失败" in answer or "超时" in answer:
            return metrics
        
        if not required_fields:
            metrics['field_coverage'] = 1.0
            metrics['missing_fields'] = []
            return metrics
        
        # 检查每个必需字段（改进：使用更宽松的匹配）
        field_coverage = []
        missing_fields = []
        
        for field in required_fields:
            # 检查字段是否在答案中，或相关词汇是否存在
            field_found = False
            
            # 字段同义词匹配
            synonyms = {
                "剂量": ["用量", "服用量", "mg", "克"],
                "证据等级": ["推荐等级", "A级", "B级", "证据"],
                "药物": ["药", "治疗", "用药"],
                "孕期": ["妊娠", "怀孕"],
                "安全性": ["安全", "风险", "禁忌"],
                "目标": ["目的", "宗旨"],
                "非药物": ["生活方式", "行为治疗"],
                "分期": ["阶段", "时期"],
                "症状": ["表现", "特征"],
                "诊断标准": ["诊断", "标准", "依据", "条件"]
            }
            
            # 检查原字段
            if field in answer:
                field_found = True
            else:
                # 检查同义词
                for synonym in synonyms.get(field, []):
                    if synonym in answer:
                        field_found = True
                        break
            
            if field_found:
                field_coverage.append(1)
            else:
                field_coverage.append(0)
                missing_fields.append(field)
        
        metrics['field_coverage'] = sum(field_coverage) / len(field_coverage)
        metrics['missing_fields'] = missing_fields
        
        return metrics
    
    def run_evaluation_suite(
        self,
        test_cases: List[Dict],
        output_file: str = "retrieval_evaluation.json"
    ) -> Dict:
        """
        运行完整的评估套件
        """
        logger.info(f"开始评估 {len(test_cases)} 个测试用例...")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n[{i}/{len(test_cases)}] 评估: {test_case['query']}")
            print(f"\n[{i}/{len(test_cases)}] 评估: {test_case['query']}")
            
            query = test_case['query']
            ground_truth = test_case.get('ground_truth', '')
            required_fields = test_case.get('required_fields', [])
            method = test_case.get('method', 'local')
            
            # 执行查询
            answer, latency, metadata = self.query_graphrag(query, method)
            
            # 评估各个维度
            accuracy = self.evaluate_accuracy(query, answer, ground_truth) if ground_truth else {}
            relevance = self.evaluate_relevance(query, answer, method)
            completeness = self.evaluate_completeness(query, answer, required_fields)
            
            # 汇总结果
            result = {
                'query': query,
                'method': method,
                'answer': answer[:200] + '...' if len(answer) > 200 else answer,
                'ground_truth': ground_truth,
                'latency': latency,
                'metadata': metadata,
                'metrics': {
                    'accuracy': accuracy,
                    'relevance': relevance,
                    'completeness': completeness
                }
            }
            
            results.append(result)
            
            # 打印简要结果
            print(f"  ⏱️  耗时: {latency:.2f}s")
            if accuracy:
                print(f"  ✅ 准确性: {accuracy.get('keyword_f1', 0):.2%}")
            print(f"  📊 相关性: {relevance.get('query_coverage', 0):.2%}")
            print(f"  📋 完整性: {completeness.get('field_coverage', 0):.2%}")
        
        # 计算总体指标
        summary = self._calculate_summary(results)
        
        # 保存结果
        full_report = {
            'summary': summary,
            'test_cases': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'root_dir': str(self.root_dir),
            'use_poetry': self.use_poetry
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估完成！结果已保存到: {output_file}")
        print(f"\n✅ 评估完成！结果已保存到: {output_file}")
        
        return full_report
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """计算总体指标"""
        summary = {
            'total_cases': len(results),
            'avg_latency': 0,
            'avg_accuracy': 0,
            'avg_relevance': 0,
            'avg_completeness': 0,
            'success_rate': 0,
            'successful_cases': 0
        }
        
        latencies = []
        accuracies = []
        relevances = []
        completenesses = []
        successes = []
        
        for result in results:
            latencies.append(result['latency'])
            
            metrics = result['metrics']
            
            # 准确性
            if 'accuracy' in metrics and metrics['accuracy']:
                acc = metrics['accuracy'].get('keyword_f1', 0)
                accuracies.append(acc)
            
            # 相关性
            if 'relevance' in metrics:
                rel = metrics['relevance'].get('query_coverage', 0)
                relevances.append(rel)
            
            # 完整性
            if 'completeness' in metrics:
                comp = metrics['completeness'].get('field_coverage', 0)
                completenesses.append(comp)
            
            # 成功率（无错误）
            has_error = result['metadata'].get('error', False)
            success = 0 if has_error else 1
            successes.append(success)
            if success:
                summary['successful_cases'] += 1
        
        # 计算平均值（避免除以0）
        summary['avg_latency'] = sum(latencies) / len(latencies) if latencies else 0
        summary['avg_accuracy'] = sum(accuracies) / len(accuracies) if accuracies else 0
        summary['avg_relevance'] = sum(relevances) / len(relevances) if relevances else 0
        summary['avg_completeness'] = sum(completenesses) / len(completenesses) if completenesses else 0
        summary['success_rate'] = sum(successes) / len(successes) if successes else 0
        
        return summary
    
    def print_report(self, report: Dict):
        """打印评估报告"""
        print("\n" + "="*80)
        print(" GraphRAG检索评估报告 ".center(80, "="))
        print("="*80)
        
        summary = report['summary']
        
        print(f"\n【总体指标】")
        print(f"  测试用例数: {summary['total_cases']}")
        print(f"  成功执行数: {summary['successful_cases']}")
        print(f"  成功率: {summary['success_rate']:.1%}")
        print(f"  平均响应时间: {summary['avg_latency']:.2f}s")
        print(f"  平均准确性: {summary['avg_accuracy']:.1%}")
        print(f"  平均相关性: {summary['avg_relevance']:.1%}")
        print(f"  平均完整性: {summary['avg_completeness']:.1%}")
        
        # 综合评分
        overall_score = (
            summary['success_rate'] * 0.2 +
            summary['avg_accuracy'] * 0.3 +
            summary['avg_relevance'] * 0.25 +
            summary['avg_completeness'] * 0.25
        ) * 100
        
        print(f"\n  🎯 综合评分: {overall_score:.1f}/100")
        
        if overall_score >= 80:
            grade = "A - 优秀"
        elif overall_score >= 70:
            grade = "B - 良好"
        elif overall_score >= 60:
            grade = "C - 中等"
        else:
            grade = "D - 需改进"
        
        print(f"  📊 等级: {grade}")
        
        print("\n【测试用例详情】")
        for i, case in enumerate(report['test_cases'], 1):
            print(f"\n  {i}. {case['query']}")
            print(f"     方法: {case['method']}")
            print(f"     耗时: {case['latency']:.2f}s")
            print(f"     执行状态: {'成功' if not case['metadata'].get('error', True) else '失败'}")
            
            metrics = case['metrics']
            if 'accuracy' in metrics and metrics['accuracy']:
                print(f"     准确性: {metrics['accuracy'].get('keyword_f1', 0):.1%}")
            print(f"     相关性: {metrics['relevance'].get('query_coverage', 0):.1%}")
            print(f"     完整性: {metrics['completeness'].get('field_coverage', 0):.1%}")
        
        print("\n" + "="*80)


def create_medical_test_cases() -> List[Dict]:
    """创建医疗指南测试用例"""
    return [
        {
            "query": "无先兆偏头痛有哪些诊断标准？",
            "ground_truth": "至少5次发作，每次4-72小时，至少满足单侧、搏动性、中重度、日常活动加重中的2项，伴恶心/呕吐或畏光/畏声",
            "required_fields": ["诊断标准", "发作次数", "持续时间", "症状特征"],
            "method": "local"
        },
        {
            "query": "药物过度使用性头痛是什么？",
            "ground_truth": "规律过度使用急性头痛药物≥3个月，每月≥10-15天，头痛频繁发作≥15天/月",
            "required_fields": ["定义", "用药时长", "发作频率", "诊断条件"],
            "method": "global"
        },
        {
            "query": "偏头痛预防性治疗的核心目标有哪些？",
            "ground_truth": "降低发作频率、减轻严重程度、避免药物过度使用性头痛",
            "required_fields": ["目标", "药物", "生活质量"],
            "method": "global"
        },
        {
            "query": "孕期偏头痛急性期首选治疗药物是什么？",
            "ground_truth": "对乙酰氨基酚",
            "required_fields": ["药物", "孕期", "安全性"],
            "method": "local"
        },
        {
            "query": "偏头痛的非药物治疗手段主要有哪些？",
            "ground_truth": "病人教育、规律作息、避免诱发因素、记录头痛日记",
            "required_fields": ["非药物", "生活方式", "管理"],
            "method": "global"
        },
    ]


def main():
    """主函数"""
    import sys
    
    # 获取GraphRAG根目录（支持命令行参数）
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # 创建评估器 - 使用poetry方式执行
    evaluator = GraphRAGRetrievalEvaluator(root_dir=root_dir, use_poetry=True)
    
    # 创建测试用例
    test_cases = create_medical_test_cases()
    
    print("="*60)
    print("GraphRAG检索评估工具 (Poetry模式)")
    print(f"GraphRAG根目录: {evaluator.root_dir}")
    print(f"测试用例数: {len(test_cases)}")
    print(f"评估维度: 准确性、相关性、完整性、响应时间")
    print("="*60)
    
    # 运行评估
    report = evaluator.run_evaluation_suite(test_cases)
    
    # 打印报告
    evaluator.print_report(report)


if __name__ == "__main__":
    main()
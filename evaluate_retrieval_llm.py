#!/usr/bin/env python3
"""
GraphRAG检索评估工具（LLM辅助版本）- 手动指定LLM配置
"""

import subprocess
import json
import time
import re
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI

# ====================== 手动配置LLM信息（从你的settings.yaml复制） ======================
# 请从你的settings.yaml中复制以下信息：
LLM_CONFIG = {
    "api_key": "sk-fd8a77d798a54e98b1b9db0f84e9ff54",       # 必填：从settings.yaml复制
    "api_base": "https://api.deepseek.com/v1", # 可选：默认deepseek地址
    "model": "deepseek-chat"                   # 可选：默认使用deepseek-chat
}
# =====================================================================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_llm.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LLMAssistedEvaluator:
    """使用LLM辅助评估检索质量（手动指定LLM配置）"""
    
    def __init__(
        self, 
        root_dir: str = ".",
        config_file: str = "./settings.yaml",
        use_poetry: bool = True,
        enable_llm_evaluation: bool = True
    ):
        """
        Args:
            root_dir: GraphRAG根目录
            config_file: settings.yaml配置文件路径
            use_poetry: 是否使用poetry run poe方式执行命令
            enable_llm_evaluation: 是否启用LLM语义评估
        """
        self.root_dir = Path(root_dir).resolve()
        self.config_file = Path(config_file).resolve()
        self.use_poetry = use_poetry
        self.enable_llm_evaluation = enable_llm_evaluation
        
        # 清理冲突环境变量
        self._clean_env_vars()
        
        # 使用手动配置的LLM信息
        self.llm_config = LLM_CONFIG
        
        # 初始化LLM客户端
        self.llm_client = None
        if self.enable_llm_evaluation and self.llm_config.get("api_key"):
            try:
                self.llm_client = OpenAI(
                    api_key=self.llm_config['api_key'],
                    base_url=self.llm_config.get('api_base', "https://api.deepseek.com/v1")
                )
                logger.info(f"✅ LLM客户端初始化成功，使用模型: {self.llm_config.get('model', 'deepseek-chat')}")
            except Exception as e:
                logger.error(f"❌ LLM客户端初始化失败: {e}")
                self.enable_llm_evaluation = False
        else:
            logger.error("❌ 未配置有效的LLM API密钥")
            self.enable_llm_evaluation = False
        
        logger.info(f"GraphRAG根目录: {self.root_dir}")
        logger.info(f"配置文件路径: {self.config_file}")
        logger.info(f"使用Poetry执行命令: {self.use_poetry}")
        logger.info(f"LLM评估启用状态: {self.enable_llm_evaluation}")
        
        # 检查配置文件
        self._check_config_file()
        
        # 检查poetry
        if self.use_poetry:
            self._check_poetry()
    
    def _clean_env_vars(self):
        """清理可能覆盖配置的环境变量"""
        env_vars = ['DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'GRAPH_RAG_API_KEY']
        for var in env_vars:
            if var in os.environ:
                logger.warning(f"清理冲突环境变量 {var}: {os.environ[var][:6]}...")
                del os.environ[var]
        logger.info("环境变量清理完成")
    
    def _check_config_file(self):
        """检查配置文件是否存在"""
        if not self.config_file.exists():
            logger.error(f"配置文件不存在: {self.config_file}")
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在")
        else:
            logger.info(f"配置文件验证通过: {self.config_file}")
    
    def _check_poetry(self):
        """检查poetry是否可用"""
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
    ) -> Tuple[str, float]:
        """执行GraphRAG查询"""
        start_time = time.time()
        
        # 构建命令
        if self.use_poetry:
            cmd = [
                "poetry", "run", "poe", "query",
                "--root", "./",
                "--config", str(self.config_file),
                "--method", method,
                "--query", query
            ]
        else:
            cmd = [
                "python", "-m", "graphrag.query",
                "--root", str(self.root_dir),
                "--config", str(self.config_file),
                "--method", method,
                "--query", query
            ]
        
        logger.info(f"执行GraphRAG查询命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.root_dir)
            )
            
            elapsed = time.time() - start_time
            
            logger.info(f"命令返回码: {result.returncode}")
            logger.info(f"标准输出长度: {len(result.stdout)}")
            logger.info(f"标准错误: {result.stderr[:200]}...")
            
            if result.returncode == 0:
                answer = result.stdout.strip()
                logger.info(f"查询成功，耗时: {elapsed:.2f}s，答案长度: {len(answer)}")
                return answer, elapsed
            else:
                error_msg = f"ERROR: {result.stderr}"
                logger.error(f"查询失败: {error_msg}")
                return error_msg, elapsed
        
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            error_msg = f"ERROR: Query timeout after {elapsed:.2f}s"
            logger.error(error_msg)
            return error_msg, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"ERROR: {str(e)}"
            logger.error(f"查询异常: {error_msg}", exc_info=True)
            return error_msg, elapsed
    
    def llm_evaluate_answer(
        self,
        query: str,
        answer: str,
        ground_truth: str
    ) -> Dict:
        """使用LLM进行语义评估"""
        # 默认评分（未启用/失败）
        default_result = {
            "accuracy": 0,
            "completeness": 0,
            "relevance": 0,
            "clarity": 0,
            "credibility": 0,
            "overall_score": 0,
            "summary": "LLM评估未启用或失败"
        }
        
        if not self.enable_llm_evaluation or not self.llm_client:
            return default_result
        
        # 答案无效
        if "ERROR" in answer or len(answer) < 10:
            logger.warning(f"答案无效，跳过LLM评估")
            return default_result
        
        # 构建评估Prompt
        eval_prompt = f"""你是一位资深的神经内科医疗专家，专注于头痛疾病的诊疗评估。
请严格按照以下标准评估医疗问答的质量：

【评估规则】
1. 准确性：回答是否符合最新的《中国偏头痛诊治指南》，无事实性错误
2. 完整性：是否覆盖问题所需的全部关键信息，无重要遗漏
3. 相关性：是否直接回答问题，无无关内容，无答非所问
4. 清晰度：表述是否专业、清晰、逻辑连贯，符合医疗规范
5. 可信度：是否引用了指南/证据来源，是否有明确的依据支持

【评估数据】
问题：{query}
标准答案：{ground_truth}
待评估回答：{answer}

【输出要求】
请返回JSON格式的评估结果，包含：
- accuracy: 0-100分（准确性）
- completeness: 0-100分（完整性）
- relevance: 0-100分（相关性）
- clarity: 0-100分（清晰度）
- credibility: 0-100分（可信度）
- overall_score: 0-100分（综合评分，取平均值）
- summary: 简短的专业评价（50-100字）

只返回JSON，不要其他内容。"""
        
        try:
            logger.info(f"调用LLM评估: {query[:20]}...")
            response = self.llm_client.chat.completions.create(
                model=self.llm_config.get('model', 'deepseek-chat'),
                messages=[
                    {"role": "system", "content": "你是专业的医疗评估专家，严格按照要求输出JSON格式的评估结果。"},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.1,
                timeout=60
            )
            
            # 解析结果
            content = response.choices[0].message.content.strip()
            content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
            
            # 解析JSON
            result = json.loads(content)
            
            # 确保数值类型正确
            for key in ['accuracy', 'completeness', 'relevance', 'clarity', 'credibility', 'overall_score']:
                if key in result:
                    result[key] = float(result[key])
            
            # 计算综合评分
            if 'overall_score' not in result:
                scores = [
                    result.get('accuracy', 0),
                    result.get('completeness', 0),
                    result.get('relevance', 0),
                    result.get('clarity', 0),
                    result.get('credibility', 0)
                ]
                result['overall_score'] = sum(scores) / len(scores)
            
            logger.info(f"LLM评估完成，综合评分: {result['overall_score']:.1f}")
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"LLM返回的内容不是有效的JSON: {content[:200]}")
            return default_result
        except Exception as e:
            logger.error(f"LLM评估失败: {str(e)}", exc_info=True)
            return default_result
    
    def run_comprehensive_evaluation(
        self,
        test_cases: List[Dict]
    ) -> Dict:
        """运行综合评估"""
        logger.info(f"开始综合评估 {len(test_cases)} 个测试用例...")
        print(f"开始综合评估 {len(test_cases)} 个测试用例...")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n[{i}/{len(test_cases)}] 评估: {test_case['query']}")
            print(f"\n[{i}/{len(test_cases)}] 评估: {test_case['query']}")
            
            query = test_case['query']
            ground_truth = test_case.get('ground_truth', '')
            method = test_case.get('method', 'local')
            
            # 执行GraphRAG查询
            answer, latency = self.query_graphrag(query, method)
            
            # 基础结果
            case_result = {
                'query': query,
                'method': method,
                'ground_truth': ground_truth,
                'answer': answer[:500] + '...' if len(answer) > 500 else answer,
                'full_answer': answer,
                'latency': latency,
                'success': not answer.startswith("ERROR"),
                'answer_length': len(answer),
                'llm_evaluation': {}
            }
            
            # 执行LLM评估
            if self.enable_llm_evaluation and case_result['success']:
                print(f"  📝 正在进行LLM语义评估...")
                llm_scores = self.llm_evaluate_answer(query, answer, ground_truth)
                case_result['llm_evaluation'] = llm_scores
                
                if 'overall_score' in llm_scores:
                    print(f"  🎯 LLM综合评分: {llm_scores['overall_score']:.1f}/100")
            else:
                print(f"  ⚠️  跳过LLM评估")
            
            # 打印基础结果
            print(f"  ⏱️  耗时: {latency:.2f}s")
            print(f"  📊 状态: {'✅ 成功' if case_result['success'] else '❌ 失败'}")
            print(f"  📝 答案长度: {case_result['answer_length']} 字符")
            
            results.append(case_result)
        
        # 汇总统计
        summary = self._calculate_comprehensive_summary(results)
        
        report = {
            'summary': summary,
            'test_cases': results,
            'config': {
                'root_dir': str(self.root_dir),
                'config_file': str(self.config_file),
                'use_poetry': self.use_poetry,
                'enable_llm_evaluation': self.enable_llm_evaluation,
                'llm_model': self.llm_config.get('model', 'unknown')
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report
    
    def _calculate_comprehensive_summary(self, results: List[Dict]) -> Dict:
        """计算综合汇总统计"""
        summary = {
            'total_cases': len(results),
            'successful_cases': sum(1 for r in results if r['success']),
            'success_rate': 0,
            'avg_latency': 0,
            'avg_answer_length': 0,
            'llm_metrics': {
                'avg_accuracy': 0,
                'avg_completeness': 0,
                'avg_relevance': 0,
                'avg_clarity': 0,
                'avg_credibility': 0,
                'avg_overall_score': 0,
                'evaluated_cases': 0
            },
            'methods': {}
        }
        
        # 基础统计
        if results:
            summary['success_rate'] = summary['successful_cases'] / len(results)
            summary['avg_latency'] = sum(r['latency'] for r in results) / len(results)
            summary['avg_answer_length'] = sum(r['answer_length'] for r in results) / len(results)
        
        # LLM评分统计
        llm_scores = {
            'accuracy': [],
            'completeness': [],
            'relevance': [],
            'clarity': [],
            'credibility': [],
            'overall_score': []
        }
        
        for result in results:
            method = result['method']
            
            # 方法统计
            if method not in summary['methods']:
                summary['methods'][method] = {
                    'count': 0,
                    'successful': 0,
                    'success_rate': 0,
                    'avg_latency': 0,
                    'avg_answer_length': 0,
                    'llm_metrics': {
                        'avg_accuracy': 0,
                        'avg_completeness': 0,
                        'avg_relevance': 0,
                        'avg_clarity': 0,
                        'avg_credibility': 0,
                        'avg_overall_score': 0,
                        'evaluated_cases': 0
                    }
                }
            
            summary['methods'][method]['count'] += 1
            summary['methods'][method]['successful'] += 1 if result['success'] else 0
            summary['methods'][method]['avg_latency'] += result['latency']
            summary['methods'][method]['avg_answer_length'] += result['answer_length']
            
            # LLM评分收集
            if result['success'] and 'llm_evaluation' in result:
                llm_eval = result['llm_evaluation']
                
                for metric in llm_scores.keys():
                    if metric in llm_eval:
                        llm_scores[metric].append(float(llm_eval[metric]))
                
                # 方法LLM评分
                for metric in ['accuracy', 'completeness', 'relevance', 'clarity', 'credibility', 'overall_score']:
                    if metric in llm_eval:
                        summary['methods'][method]['llm_metrics'][f'avg_{metric}'] += float(llm_eval[metric])
                
                summary['methods'][method]['llm_metrics']['evaluated_cases'] += 1
        
        # 方法统计平均值
        for method, stats in summary['methods'].items():
            if stats['count'] > 0:
                stats['success_rate'] = stats['successful'] / stats['count']
                stats['avg_latency'] /= stats['count']
                stats['avg_answer_length'] /= stats['count']
                
                # LLM平均分
                llm_metrics = stats['llm_metrics']
                if llm_metrics['evaluated_cases'] > 0:
                    for metric in ['accuracy', 'completeness', 'relevance', 'clarity', 'credibility', 'overall_score']:
                        key = f'avg_{metric}'
                        llm_metrics[key] /= llm_metrics['evaluated_cases']
        
        # 总体LLM平均分
        for metric, scores in llm_scores.items():
            if scores:
                summary['llm_metrics'][f'avg_{metric}'] = sum(scores) / len(scores)
        
        summary['llm_metrics']['evaluated_cases'] = len(llm_scores['overall_score'])
        
        return summary
    
    def print_comprehensive_report(self, report: Dict):
        """打印综合评估报告"""
        print("\n" + "="*80)
        print(" GraphRAG检索综合评估报告（LLM语义版） ".center(80, "="))
        print("="*80)
        
        summary = report['summary']
        config = report['config']
        
        print(f"\n【配置信息】")
        print(f"  GraphRAG根目录: {config['root_dir']}")
        print(f"  配置文件: {config['config_file']}")
        print(f"  LLM模型: {config['llm_model']}")
        print(f"  LLM评估: {'✅ 已启用' if config['enable_llm_evaluation'] else '❌ 未启用'}")
        
        print(f"\n【总体统计】")
        print(f"  测试用例总数: {summary['total_cases']}")
        print(f"  成功执行数: {summary['successful_cases']}")
        print(f"  总体成功率: {summary['success_rate']:.1%}")
        print(f"  平均响应时间: {summary['avg_latency']:.2f}s")
        print(f"  平均答案长度: {summary['avg_answer_length']:.0f} 字符")
        
        if config['enable_llm_evaluation'] and summary['llm_metrics']['evaluated_cases'] > 0:
            print(f"\n【LLM语义评估总体评分】")
            print(f"  评估用例数: {summary['llm_metrics']['evaluated_cases']}")
            print(f"  平均准确性: {summary['llm_metrics']['avg_accuracy']:.1f}/100")
            print(f"  平均完整性: {summary['llm_metrics']['avg_completeness']:.1f}/100")
            print(f"  平均相关性: {summary['llm_metrics']['avg_relevance']:.1f}/100")
            print(f"  平均清晰度: {summary['llm_metrics']['avg_clarity']:.1f}/100")
            print(f"  平均可信度: {summary['llm_metrics']['avg_credibility']:.1f}/100")
            print(f"  综合平均分: {summary['llm_metrics']['avg_overall_score']:.1f}/100")
        
        print(f"\n【各方法表现】")
        for method, stats in summary['methods'].items():
            print(f"\n  {method.upper()}:")
            print(f"    查询数量: {stats['count']}")
            print(f"    成功率: {stats['success_rate']:.1%}")
            print(f"    平均响应时间: {stats['avg_latency']:.2f}s")
            print(f"    平均答案长度: {stats['avg_answer_length']:.0f} 字符")
            
            if config['enable_llm_evaluation'] and stats['llm_metrics']['evaluated_cases'] > 0:
                llm = stats['llm_metrics']
                print(f"    LLM评分（平均）:")
                print(f"      准确性: {llm['avg_accuracy']:.1f}/100")
                print(f"      完整性: {llm['avg_completeness']:.1f}/100")
                print(f"      相关性: {llm['avg_relevance']:.1f}/100")
                print(f"      清晰度: {llm['avg_clarity']:.1f}/100")
                print(f"      可信度: {llm['avg_credibility']:.1f}/100")
                print(f"      综合分: {llm['avg_overall_score']:.1f}/100")
        
        # 详细案例
        print(f"\n【测试用例详情】")
        for i, case in enumerate(report['test_cases'], 1):
            print(f"\n  {i}. {case['query']}")
            print(f"     方法: {case['method']}")
            print(f"     耗时: {case['latency']:.2f}s")
            print(f"     状态: {'成功' if case['success'] else '失败'}")
            print(f"     答案长度: {case['answer_length']} 字符")
            
            # LLM评分
            if config['enable_llm_evaluation'] and case['success'] and 'llm_evaluation' in case:
                llm = case['llm_evaluation']
                print(f"     LLM评分:")
                print(f"       准确性: {llm.get('accuracy', 0):.1f}/100")
                print(f"       完整性: {llm.get('completeness', 0):.1f}/100")
                print(f"       相关性: {llm.get('relevance', 0):.1f}/100")
                print(f"       清晰度: {llm.get('clarity', 0):.1f}/100")
                print(f"       可信度: {llm.get('credibility', 0):.1f}/100")
                print(f"       综合分: {llm.get('overall_score', 0):.1f}/100")
                if 'summary' in llm:
                    print(f"       评价: {llm['summary'][:100]}...")
            
            print(f"     答案预览: {case['answer'][:100]}...")
        
        print("\n" + "="*80)
    
    def save_report(self, report: Dict, output_file: str):
        """保存报告"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"报告已保存到: {output_file}")
            print(f"✅ 详细报告已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
            print(f"❌ 保存报告失败: {str(e)}")


def create_test_cases() -> List[Dict]:
    """创建测试用例"""
    return [
        {
            "query": "无先兆偏头痛有哪些诊断标准？",
            "ground_truth": "至少5次发作，每次4-72小时，至少满足单侧、搏动性、中重度、日常活动加重中的2项，伴恶心/呕吐或畏光/畏声，且不能归因于其他疾病。",
            "method": "local"
        },
        {
            "query": "药物过度使用性头痛是什么？",
            "ground_truth": "规律过度使用急性头痛药物≥3个月，每月≥10-15天，头痛频繁发作≥15天/月，属于继发性头痛，停用过度使用药物后症状可缓解。",
            "method": "local"
        },
        {
            "query": "偏头痛预防性治疗的核心目标有哪些？",
            "ground_truth": "降低发作频率、减轻发作严重程度、缩短发作持续时间、减少急性期药物使用、改善患者生活质量，避免发展为慢性偏头痛。",
            "method": "local"
        },
        {
            "query": "孕期偏头痛急性期首选治疗药物是什么？",
            "ground_truth": "对乙酰氨基酚是孕期偏头痛急性期首选治疗药物，在孕期各阶段使用均相对安全，避免使用曲坦类、NSAIDs等可能有风险的药物。",
            "method": "local"
        },
        {
            "query": "偏头痛的非药物治疗手段主要有哪些？",
            "ground_truth": "包括病人教育、规律作息、避免诱发因素、认知行为治疗、放松训练、生物反馈治疗、针灸、适度有氧运动等，是偏头痛管理的基础。",
            "method": "local"
        }
    ]


def main():
    """主函数"""
    import sys
    
    # 获取参数
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    config_file = sys.argv[2] if len(sys.argv) > 2 else "./settings.yaml"
    
    # 检查API配置
    if not LLM_CONFIG.get("api_key") or LLM_CONFIG["api_key"] == "你的deepseek API密钥":
        print("❌ 请先在脚本开头的LLM_CONFIG中填写有效的DeepSeek API密钥！")
        return
    
    # 创建评估器
    try:
        evaluator = LLMAssistedEvaluator(
            root_dir=root_dir,
            config_file=config_file,
            use_poetry=True,
            enable_llm_evaluation=True
        )
    except FileNotFoundError as e:
        print(f"❌ 配置文件错误: {e}")
        return
    
    # 创建测试用例
    test_cases = create_test_cases()
    
    print("="*60)
    print("GraphRAG检索综合评估工具（LLM语义版）")
    print(f"根目录: {evaluator.root_dir}")
    print(f"配置文件: {evaluator.config_file}")
    print(f"LLM模型: {evaluator.llm_config.get('model', 'unknown')}")
    print(f"测试用例数: {len(test_cases)}")
    print("="*60)
    
    # 运行评估
    report = evaluator.run_comprehensive_evaluation(test_cases)
    
    # 打印报告
    evaluator.print_comprehensive_report(report)
    
    # 保存报告
    evaluator.save_report(report, "retrieval_llm_evaluation.json")


if __name__ == "__main__":
    main()
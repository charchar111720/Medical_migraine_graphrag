#!/usr/bin/env python3
"""
GraphRAG提取结果量化评估工具

评估维度：
1. 实体质量（覆盖率、准确性、一致性）
2. 关系质量（完整性、强度分布）
3. 社区质量（规模、层次）
4. 整体指标（去噪效果、命名规范）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import datetime  # 新增：导入日期模块

class GraphRAGEvaluator:
    """GraphRAG提取结果评估器"""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: GraphRAG输出目录（包含artifacts）
        """
        self.output_dir = Path(output_dir)
        self.entities = pd.read_parquet(self.output_dir / "entities.parquet")
        self.relationships = pd.read_parquet(self.output_dir / "relationships.parquet")
        self.communities = pd.read_parquet(self.output_dir / "communities.parquet")
        self.text_units = pd.read_parquet(self.output_dir / "text_units.parquet")
    
    def _convert_numpy_types(self, data):
        """递归将NumPy类型转为Python原生类型（解决JSON序列化问题）"""
        if isinstance(data, dict):
            return {k: self._convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_types(i) for i in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
        
    def evaluate_entities(self) -> Dict:
        """评估实体质量"""
        metrics = {}
        
        # 1. 覆盖率指标
        metrics['total_entities'] = len(self.entities)
        metrics['entity_types'] = self.entities['type'].nunique()
        metrics['type_distribution'] = self.entities['type'].value_counts().to_dict()
        
        # 2. 准确性指标 - 检查不应出现的类型
        forbidden_types = ['PERSON', 'ORGANIZATION', 'LOCATION', 'GEO']
        forbidden_count = self.entities[self.entities['type'].isin(forbidden_types)].shape[0]
        metrics['forbidden_entities'] = forbidden_count
        metrics['forbidden_ratio'] = forbidden_count / len(self.entities) if len(self.entities) > 0 else 0
        
        # 3. 一致性指标 - 检查命名规范
        evidence_entities = self.entities[self.entities['type'] == 'EVIDENCE_LEVEL']
        rec_entities = self.entities[self.entities['type'] == 'RECOMMENDATION_GRADE']
        
        # 证据等级标准形式
        standard_evidence = {'A级', 'B级', 'C级', 'D级', 'Ⅰ级', 'Ⅱ级', 'Ⅲ级', 'Ⅳ级', '高', '中', '低', '极低'}
        non_standard_evidence = set(evidence_entities['title'].unique()) - standard_evidence
        
        # 推荐强度标准形式
        standard_rec = {'强推荐', '弱推荐', '1级', '2级', '强', '弱'}
        non_standard_rec = set(rec_entities['title'].unique()) - standard_rec
        
        metrics['evidence_level_count'] = len(evidence_entities)
        metrics['evidence_standard_ratio'] = (len(evidence_entities) - len(non_standard_evidence)) / len(evidence_entities) if len(evidence_entities) > 0 else 0
        metrics['non_standard_evidence'] = list(non_standard_evidence)
        
        metrics['recommendation_count'] = len(rec_entities)
        metrics['recommendation_standard_ratio'] = (len(rec_entities) - len(non_standard_rec)) / len(rec_entities) if len(rec_entities) > 0 else 0
        metrics['non_standard_recommendation'] = list(non_standard_rec)
        
        # 4. 描述质量
        self.entities['desc_len'] = self.entities['description'].str.len()
        metrics['avg_description_length'] = self.entities['desc_len'].mean()
        metrics['max_description_length'] = self.entities['desc_len'].max()
        metrics['over_length_count'] = (self.entities['desc_len'] > 50).sum()
        metrics['over_length_ratio'] = (self.entities['desc_len'] > 50).sum() / len(self.entities)
        
        # 5. 度中心性（重要性）
        metrics['avg_degree'] = self.entities['degree'].mean() if 'degree' in self.entities.columns else 0
        metrics['max_degree'] = self.entities['degree'].max() if 'degree' in self.entities.columns else 0
        
        # 找出度最高的实体
        if 'degree' in self.entities.columns:
            top_entities = self.entities.nlargest(5, 'degree')[['title', 'type', 'degree']]
            metrics['top_entities'] = top_entities.to_dict('records')
        
        return metrics
    
    def evaluate_relationships(self) -> Dict:
        """评估关系质量"""
        metrics = {}
        
        # 1. 完整性指标
        metrics['total_relationships'] = len(self.relationships)
        metrics['relationships_per_entity'] = len(self.relationships) / len(self.entities) if len(self.entities) > 0 else 0
        
        # 2. 强度分布
        if 'weight' in self.relationships.columns:
            metrics['avg_weight'] = self.relationships['weight'].mean()
            metrics['min_weight'] = self.relationships['weight'].min()
            metrics['max_weight'] = self.relationships['weight'].max()
            metrics['weight_std'] = self.relationships['weight'].std()
            
            # 强度分布
            metrics['weight_distribution'] = {
                'weak (1-3)': (self.relationships['weight'] <= 3).sum(),
                'medium (4-7)': ((self.relationships['weight'] > 3) & (self.relationships['weight'] <= 7)).sum(),
                'strong (8-10)': (self.relationships['weight'] > 7).sum()
            }
        
        # 3. 关系多样性
        if 'description' in self.relationships.columns:
            # 通过描述推断关系类型
            rel_types = self.relationships['description'].apply(self._infer_relation_type)
            metrics['inferred_relation_types'] = rel_types.value_counts().to_dict()
        
        # 4. 孤立实体检查
        entity_ids = set(self.entities['title'].unique())
        source_entities = set(self.relationships['source'].unique())
        target_entities = set(self.relationships['target'].unique())
        connected_entities = source_entities | target_entities
        isolated_entities = entity_ids - connected_entities
        
        metrics['isolated_entities_count'] = len(isolated_entities)
        metrics['isolated_ratio'] = len(isolated_entities) / len(entity_ids) if len(entity_ids) > 0 else 0
        
        return metrics
    
    def _infer_relation_type(self, description: str) -> str:
        """从关系描述推断关系类型"""
        if not description:
            return 'UNKNOWN'
        
        desc_lower = description.lower()
        if '属于' in desc_lower or 'belongs' in desc_lower:
            return 'BELONGS_TO'
        elif '治疗' in desc_lower or 'treats' in desc_lower:
            return 'TREATS'
        elif '预防' in desc_lower or 'prevents' in desc_lower:
            return 'PREVENTS'
        elif '证据' in desc_lower or 'evidence' in desc_lower:
            return 'HAS_EVIDENCE'
        elif '推荐' in desc_lower or 'recommendation' in desc_lower:
            return 'HAS_STRENGTH'
        else:
            return 'OTHER'
    
    def evaluate_communities(self) -> Dict:
        """评估社区质量"""
        metrics = {}
        
        # 1. 规模指标
        metrics['total_communities'] = len(self.communities)
        metrics['level_distribution'] = self.communities['level'].value_counts().to_dict()
        
        # 2. 社区大小
        if 'size' in self.communities.columns:
            metrics['avg_community_size'] = self.communities['size'].mean()
            metrics['max_community_size'] = self.communities['size'].max()
            metrics['min_community_size'] = self.communities['size'].min()
        
        # 3. 层次结构
        max_level = self.communities['level'].max()
        metrics['max_hierarchy_level'] = max_level
        metrics['hierarchy_depth'] = max_level + 1
        
        return metrics
    
    def evaluate_text_units(self) -> Dict:
        """评估文本块质量"""
        metrics = {}
        
        # 1. 块数量和大小
        metrics['total_chunks'] = len(self.text_units)
        
        if 'n_tokens' in self.text_units.columns:
            metrics['avg_chunk_size'] = self.text_units['n_tokens'].mean()
            metrics['max_chunk_size'] = self.text_units['n_tokens'].max()
            metrics['min_chunk_size'] = self.text_units['n_tokens'].min()
            metrics['chunk_size_std'] = self.text_units['n_tokens'].std()
        
        return metrics
    
    def calculate_overall_score(self, entity_metrics: Dict, rel_metrics: Dict) -> Dict:
        """计算综合评分（0-100）"""
        scores = {}
        
        # 1. 实体质量得分（30分）
        entity_score = 0
        # 无禁止类型：10分
        if entity_metrics['forbidden_ratio'] == 0:
            entity_score += 10
        else:
            entity_score += max(0, 10 - entity_metrics['forbidden_ratio'] * 100)
        
        # 命名规范：10分
        evidence_score = entity_metrics.get('evidence_standard_ratio', 0) * 5
        rec_score = entity_metrics.get('recommendation_standard_ratio', 0) * 5
        entity_score += evidence_score + rec_score
        
        # 描述长度合理：10分
        if entity_metrics['over_length_ratio'] < 0.1:
            entity_score += 10
        elif entity_metrics['over_length_ratio'] < 0.3:
            entity_score += 7
        else:
            entity_score += max(0, 10 - entity_metrics['over_length_ratio'] * 20)
        
        scores['entity_quality'] = round(entity_score, 2)
        
        # 2. 关系质量得分（30分）
        rel_score = 0
        # 关系密度：10分
        if rel_metrics['relationships_per_entity'] >= 2:
            rel_score += 10
        else:
            rel_score += rel_metrics['relationships_per_entity'] * 5
        
        # 强度分布合理：10分
        if 'weight_distribution' in rel_metrics:
            strong_ratio = rel_metrics['weight_distribution']['strong (8-10)'] / rel_metrics['total_relationships']
            if 0.3 <= strong_ratio <= 0.7:
                rel_score += 10
            else:
                rel_score += max(0, 10 - abs(strong_ratio - 0.5) * 20)
        
        # 孤立实体少：10分
        if rel_metrics['isolated_ratio'] < 0.1:
            rel_score += 10
        elif rel_metrics['isolated_ratio'] < 0.3:
            rel_score += 7
        else:
            rel_score += max(0, 10 - rel_metrics['isolated_ratio'] * 20)
        
        scores['relationship_quality'] = round(rel_score, 2)
        
        # 3. 覆盖率得分（20分）
        coverage_score = 0
        # 实体类型多样性：10分
        expected_types = 13  # 医疗指南应有13种类型
        coverage_score += min(10, entity_metrics['entity_types'] / expected_types * 10)
        
        # 实体数量合理：10分
        if 200 <= entity_metrics['total_entities'] <= 500:
            coverage_score += 10
        elif 100 <= entity_metrics['total_entities'] < 200:
            coverage_score += 7
        else:
            coverage_score += 5
        
        scores['coverage'] = round(coverage_score, 2)
        
        # 4. 结构质量得分（20分）
        # （基于社区和文本块，简化评分）
        scores['structure_quality'] = 15.0  # 默认给15分
        
        # 总分
        scores['total'] = round(sum(scores.values()), 2)
        
        return scores
    
    def generate_report(self) -> Dict:
        """生成完整评估报告"""
        print("开始评估GraphRAG提取结果...")
        
        entity_metrics = self.evaluate_entities()
        rel_metrics = self.evaluate_relationships()
        community_metrics = self.evaluate_communities()
        chunk_metrics = self.evaluate_text_units()
        scores = self.calculate_overall_score(entity_metrics, rel_metrics)
        
        report = {
            'summary': {
                'total_score': scores['total'],
                'scores_breakdown': scores,
                'grade': self._get_grade(scores['total'])
            },
            'entities': entity_metrics,
            'relationships': rel_metrics,
            'communities': community_metrics,
            'text_units': chunk_metrics
        }
        
        # 关键修改：转换所有NumPy类型为Python原生类型
        report = self._convert_numpy_types(report)
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """根据分数给出等级"""
        if score >= 90:
            return 'A - 优秀'
        elif score >= 80:
            return 'B - 良好'
        elif score >= 70:
            return 'C - 中等'
        elif score >= 60:
            return 'D - 及格'
        else:
            return 'F - 不及格'
    
    def print_report(self, report: Dict):
        """打印评估报告"""
        print("\n" + "="*80)
        print(" GraphRAG提取结果评估报告 ".center(80, "="))
        print("="*80)
        
        # 总分
        summary = report['summary']
        print(f"\n【综合评分】{summary['total_score']:.2f}/100  等级：{summary['grade']}")
        
        # 分项得分
        print("\n【分项得分】")
        for key, value in summary['scores_breakdown'].items():
            if key != 'total':
                print(f"  - {key}: {value:.2f}")
        
        # 实体质量
        print("\n【实体质量】")
        ent = report['entities']
        print(f"  总实体数: {ent['total_entities']}")
        print(f"  实体类型数: {ent['entity_types']}")
        print(f"  禁止类型数: {ent['forbidden_entities']} ({ent['forbidden_ratio']*100:.1f}%)")
        print(f"  证据等级规范率: {ent['evidence_standard_ratio']*100:.1f}%")
        print(f"  推荐强度规范率: {ent['recommendation_standard_ratio']*100:.1f}%")
        print(f"  平均描述长度: {ent['avg_description_length']:.0f} 字符")
        print(f"  超长描述数: {ent['over_length_count']} ({ent['over_length_ratio']*100:.1f}%)")
        
        if ent['non_standard_evidence']:
            print(f"  ⚠️  非标准证据等级: {ent['non_standard_evidence']}")
        if ent['non_standard_recommendation']:
            print(f"  ⚠️  非标准推荐强度: {ent['non_standard_recommendation']}")
        
        # 关系质量
        print("\n【关系质量】")
        rel = report['relationships']
        print(f"  总关系数: {rel['total_relationships']}")
        print(f"  关系密度: {rel['relationships_per_entity']:.2f} (关系/实体)")
        if 'avg_weight' in rel:
            print(f"  平均权重: {rel['avg_weight']:.2f}")
            print(f"  权重分布: 弱({rel['weight_distribution']['weak (1-3)']}) "
                  f"中({rel['weight_distribution']['medium (4-7)']}) "
                  f"强({rel['weight_distribution']['strong (8-10)']})")
        print(f"  孤立实体数: {rel['isolated_entities_count']} ({rel['isolated_ratio']*100:.1f}%)")
        
        # 社区质量
        print("\n【社区质量】")
        comm = report['communities']
        print(f"  总社区数: {comm['total_communities']}")
        print(f"  层次深度: {comm['hierarchy_depth']}")
        
        # 改进建议
        print("\n【改进建议】")
        self._print_suggestions(report)
        
        print("\n" + "="*80)
    
    def _print_suggestions(self, report: Dict):
        """打印改进建议"""
        suggestions = []
        
        ent = report['entities']
        rel = report['relationships']
        
        if ent['forbidden_ratio'] > 0:
            suggestions.append(f"❗ 存在{ent['forbidden_entities']}个不应提取的实体类型，建议加强Prompt约束")
        
        if ent['evidence_standard_ratio'] < 0.9:
            suggestions.append(f"⚠️  证据等级命名不规范（{ent['evidence_standard_ratio']*100:.0f}%），建议明确标准形式列表")
        
        if ent['recommendation_standard_ratio'] < 0.9:
            suggestions.append(f"⚠️  推荐强度命名不规范（{ent['recommendation_standard_ratio']*100:.0f}%），建议明确标准形式列表")
        
        if ent['over_length_ratio'] > 0.3:
            suggestions.append(f"⚠️  {ent['over_length_ratio']*100:.0f}%的描述过长，建议限制描述长度为50字")
        
        if rel['relationships_per_entity'] < 1.5:
            suggestions.append(f"⚠️  关系密度偏低（{rel['relationships_per_entity']:.2f}），可能提取不完整")
        
        if rel['isolated_ratio'] > 0.2:
            suggestions.append(f"⚠️  {rel['isolated_ratio']*100:.0f}%的实体孤立，建议检查关系提取逻辑")
        
        if not suggestions:
            suggestions.append("✅ 提取质量良好，继续保持！")
        
        for i, sug in enumerate(suggestions, 1):
            print(f"  {i}. {sug}")


def main():
    """主函数"""
    import sys
    from datetime import datetime
    
    if len(sys.argv) < 2:
        print("使用方法: python evaluate_graphrag.py <output_dir>")
        print("示例: python evaluate_graphrag.py output/20260228_120000/artifacts")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    evaluator = GraphRAGEvaluator(output_dir)
    report = evaluator.generate_report()
    
    # 打印报告
    evaluator.print_report(report)
    
    # 核心修改：生成带日期的文件名
    today_str = datetime.now().strftime("%Y%m%d")
    output_file = Path(output_dir).parent / f"evaluation_report_{today_str}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存到: {output_file}")


if __name__ == "__main__":
    main()
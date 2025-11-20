#!/usr/bin/env python3
"""
增强版数据验证脚本，用于对原始数据和爬虫获取的数据进行全面验证
检查数据的完整性、格式规范性、数据准确性及结构一致性
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Any, Optional

class EnhancedDataValidator:
    """增强版数据验证器"""
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """初始化验证器
        
        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.validation_results = {}
        self.summary_stats = {
            'total_files': 0,
            'files_with_issues': 0,
            'files_with_missing_data': 0,
            'files_with_format_issues': 0,
            'files_with_accuracy_issues': 0,
            'files_with_structure_issues': 0
        }
        
    def validate_all_files(self, file_pattern: str = "*.csv") -> Dict:
        """验证所有匹配模式的文件
        
        Args:
            file_pattern: 文件匹配模式
            
        Returns:
            验证结果字典
        """
        # 递归搜索所有子目录中的CSV文件
        csv_files = list(self.data_dir.rglob(file_pattern))
        self.summary_stats['total_files'] = len(csv_files)
        
        print(f"开始验证 {len(csv_files)} 个CSV文件...")
        
        for file_path in csv_files:
            print(f"正在验证: {file_path.relative_to(self.data_dir)}")
            file_result = self._validate_single_file(file_path)
            # 使用相对路径作为键，避免重复文件名冲突
            relative_path = str(file_path.relative_to(self.data_dir))
            self.validation_results[relative_path] = file_result
            
            # 更新统计信息
            if file_result['has_issues']:
                self.summary_stats['files_with_issues'] += 1
            if file_result['missing_data_percentage'] > 0:
                self.summary_stats['files_with_missing_data'] += 1
            if file_result['format_issues']:
                self.summary_stats['files_with_format_issues'] += 1
            if file_result['accuracy_issues']:
                self.summary_stats['files_with_accuracy_issues'] += 1
            if file_result['structure_issues']:
                self.summary_stats['files_with_structure_issues'] += 1
        
        return self.validation_results
    
    def _validate_single_file(self, file_path: Path) -> Dict:
        """验证单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件验证结果
        """
        result = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'validation_time': datetime.now().isoformat(),
            'has_issues': False,
            'missing_data_percentage': 0,
            'format_issues': [],
            'accuracy_issues': [],
            'structure_issues': [],
            'recommendations': [],
            'overall_score': 100  # 满分100
        }
        
        try:
            # 尝试读取文件
            df = pd.read_csv(file_path)
            result['row_count'] = len(df)
            result['column_count'] = len(df.columns)
            result['columns'] = list(df.columns)
            
            # 1. 验证数据完整性
            missing_data = self._check_missing_data(df)
            result.update(missing_data)
            
            # 2. 验证格式规范性
            format_issues = self._check_format_compliance(df, file_path.name)
            result['format_issues'] = format_issues
            
            # 3. 验证数据准确性
            accuracy_issues = self._check_data_accuracy(df, file_path.name)
            result['accuracy_issues'] = accuracy_issues
            
            # 4. 验证结构一致性
            structure_issues = self._check_structure_consistency(df, file_path.name)
            result['structure_issues'] = structure_issues
            
            # 生成建议
            recommendations = self._generate_recommendations(result)
            result['recommendations'] = recommendations
            
            # 计算总体评分
            overall_score = self._calculate_overall_score(result)
            result['overall_score'] = overall_score
            
            # 判断是否有问题
            result['has_issues'] = (
                result['missing_data_percentage'] > 0 or
                len(result['format_issues']) > 0 or
                len(result['accuracy_issues']) > 0 or
                len(result['structure_issues']) > 0
            )
            
        except Exception as e:
            result['has_issues'] = True
            result['structure_issues'].append(f"文件读取失败: {str(e)}")
            result['overall_score'] = 0
        
        return result
    
    def _check_missing_data(self, df: pd.DataFrame) -> Dict:
        """检查缺失数据
        
        Args:
            df: 数据框
            
        Returns:
            缺失数据检查结果
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # 按列检查缺失数据
        column_missing = {}
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            col_missing_pct = (col_missing / len(df)) * 100 if len(df) > 0 else 0
            column_missing[col] = {
                'count': int(col_missing),
                'percentage': round(col_missing_pct, 2)
            }
        
        return {
            'missing_data_percentage': round(missing_percentage, 2),
            'column_missing_data': column_missing
        }
    
    def _check_format_compliance(self, df: pd.DataFrame, filename: str) -> List[Dict]:
        """检查格式规范性
        
        Args:
            df: 数据框
            filename: 文件名
            
        Returns:
            格式问题列表
        """
        issues = []
        
        # 检查列名规范性
        for col in df.columns:
            # 确保列名是字符串类型
            col_str = str(col)
            
            # 检查列名是否包含特殊字符或空格
            if re.search(r'[^a-zA-Z0-9_]', col_str):
                issues.append({
                    'type': 'column_name_format',
                    'column': col_str,
                    'description': f"列名 '{col_str}' 包含特殊字符或空格，建议使用下划线",
                    'severity': 'medium'
                })
            
            # 检查列名是否过长
            if len(col_str) > 50:
                issues.append({
                    'type': 'column_name_length',
                    'column': col_str,
                    'description': f"列名 '{col_str}' 过长({len(col_str)}字符)，建议缩短",
                    'severity': 'low'
                })
        
        # 检查数据类型
        for col in df.columns:
            dtype = df[col].dtype
            # 检查是否应该是数值型但存储为字符串
            if dtype == 'object':
                sample_values = df[col].dropna().head(10).astype(str)
                if sample_values.str.match(r'^[\d,.-]+$').all():
                    try:
                        pd.to_numeric(df[col].str.replace(',', ''), errors='raise')
                        issues.append({
                            'type': 'data_type_mismatch',
                            'column': col,
                            'description': f"列 '{col}' 应为数值型但存储为字符串",
                            'severity': 'high'
                        })
                    except:
                        pass
        
        # 检查日期格式
        date_columns = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['date', 'time', 'year', 'month', 'day'])]
        for col in date_columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col], errors='raise')
                except:
                    issues.append({
                        'type': 'date_format',
                        'column': col,
                        'description': f"日期列 '{col}' 格式不规范",
                        'severity': 'high'
                    })
        
        return issues
    
    def _check_data_accuracy(self, df: pd.DataFrame, filename: str) -> List[Dict]:
        """检查数据准确性
        
        Args:
            df: 数据框
            filename: 文件名
            
        Returns:
            数据准确性问题列表
        """
        issues = []
        
        # 检查数值列的异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df)) * 100
                issues.append({
                    'type': 'outliers',
                    'column': col,
                    'description': f"列 '{col}' 检测到 {len(outliers)} 个异常值 ({outlier_percentage:.2f}%)",
                    'severity': 'medium' if outlier_percentage < 5 else 'high',
                    'outlier_count': len(outliers),
                    'outlier_percentage': round(outlier_percentage, 2)
                })
        
        # 检查负值是否合理
        for col in numeric_columns:
            if 'value' in col.lower() or 'price' in col.lower() or 'amount' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    negative_percentage = (negative_count / len(df)) * 100
                    issues.append({
                        'type': 'negative_values',
                        'column': col,
                        'description': f"值列 '{col}' 包含 {negative_count} 个负值 ({negative_percentage:.2f}%)",
                        'severity': 'medium',
                        'negative_count': int(negative_count),
                        'negative_percentage': round(negative_percentage, 2)
                    })
        
        # 检查重复行
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            duplicate_percentage = (duplicate_rows / len(df)) * 100
            issues.append({
                'type': 'duplicate_rows',
                'description': f"检测到 {duplicate_rows} 行重复数据 ({duplicate_percentage:.2f}%)",
                'severity': 'medium',
                'duplicate_count': int(duplicate_rows),
                'duplicate_percentage': round(duplicate_percentage, 2)
            })
        
        # 特定文件的准确性检查
        if 'china_imports_soybeans' in filename.lower():
            # 检查HS编码是否为1201
            if 'hs_code' in df.columns:
                invalid_hs = df[~df['hs_code'].astype(str).str.startswith('1201', na=False)]
                if len(invalid_hs) > 0:
                    issues.append({
                        'type': 'invalid_hs_code',
                        'description': f"检测到 {len(invalid_hs)} 行非1201开头的HS编码",
                        'severity': 'high',
                        'invalid_count': len(invalid_hs)
                    })
        
        return issues
    
    def _check_structure_consistency(self, df: pd.DataFrame, filename: str) -> List[Dict]:
        """检查结构一致性
        
        Args:
            df: 数据框
            filename: 文件名
            
        Returns:
            结构一致性问题列表
        """
        issues = []
        
        # 检查是否有唯一标识符
        column_names_lower = [col.lower() for col in df.columns]
        if 'id' not in column_names_lower and 'index' not in column_names_lower:
            # 检查是否有组合键
            potential_keys = ['year', 'date', 'period']
            existing_keys = [key for key in potential_keys if key in df.columns]
            if not existing_keys:
                issues.append({
                    'type': 'missing_identifier',
                    'description': "文件缺少唯一标识符或时间戳列",
                    'severity': 'medium'
                })
        
        # 检查时间序列数据的连续性
        time_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['year', 'date', 'time', 'period'])]
        for col in time_columns:
            if df[col].dtype == 'object':
                try:
                    time_series = pd.to_datetime(df[col], errors='coerce')
                    if time_series.isnull().any():
                        null_count = time_series.isnull().sum()
                        issues.append({
                            'type': 'invalid_time_format',
                            'column': col,
                            'description': f"时间列 '{col}' 包含 {null_count} 个无效时间格式",
                            'severity': 'high',
                            'invalid_count': int(null_count)
                        })
                except:
                    pass
            elif df[col].dtype in ['int64', 'float64']:
                # 检查年份是否在合理范围内
                if 'year' in col.lower():
                    invalid_years = df[(df[col] < 2000) | (df[col] > 2030)]
                    if len(invalid_years) > 0:
                        issues.append({
                            'type': 'invalid_year_range',
                            'column': col,
                            'description': f"年份列 '{col}' 包含 {len(invalid_years)} 个不合理年份",
                            'severity': 'high',
                            'invalid_count': len(invalid_years)
                        })
        
        # 检查数据结构是否符合预期
        expected_structures = {
            'china_imports_soybeans': ['year', 'importer', 'exporter', 'import_value_usd', 'import_quantity_tonnes', 'hs_code'],
            'fred': ['series_id', 'date', 'year', 'value']
        }
        
        if 'china_imports_soybeans' in filename.lower():
            expected_cols = expected_structures['china_imports_soybeans']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                issues.append({
                    'type': 'missing_expected_columns',
                    'description': f"缺少预期列: {', '.join(missing_cols)}",
                    'severity': 'high',
                    'missing_columns': missing_cols
                })
        
        if 'fred' in filename.lower():
            expected_cols = expected_structures['fred']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                issues.append({
                    'type': 'missing_expected_columns',
                    'description': f"缺少预期列: {', '.join(missing_cols)}",
                    'severity': 'high',
                    'missing_columns': missing_cols
                })
        
        return issues
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """生成改进建议
        
        Args:
            result: 验证结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于缺失数据的建议
        if result['missing_data_percentage'] > 0:
            if result['missing_data_percentage'] < 5:
                recommendations.append("少量缺失数据，建议使用插值或均值填充")
            elif result['missing_data_percentage'] < 20:
                recommendations.append("中等量缺失数据，建议使用更复杂的插值方法或模型预测")
            else:
                recommendations.append("大量缺失数据，建议重新收集数据或考虑删除相关记录")
        
        # 基于格式问题的建议
        for issue in result['format_issues']:
            if issue['type'] == 'column_name_format':
                recommendations.append(f"标准化列名 '{issue['column']}'，使用下划线替换特殊字符")
            elif issue['type'] == 'data_type_mismatch':
                recommendations.append(f"转换列 '{issue['column']}' 为正确的数值类型")
            elif issue['type'] == 'date_format':
                recommendations.append(f"标准化列 '{issue['column']}' 的日期格式为ISO 8601")
        
        # 基于准确性问题的建议
        for issue in result['accuracy_issues']:
            if issue['type'] == 'outliers':
                recommendations.append(f"检查列 '{issue['column']}' 中的异常值，确认是否为数据录入错误")
            elif issue['type'] == 'negative_values':
                recommendations.append(f"检查列 '{issue['column']}' 中的负值，确认是否合理")
            elif issue['type'] == 'duplicate_rows':
                recommendations.append("删除重复行或确认重复数据的合理性")
        
        # 基于结构问题的建议
        for issue in result['structure_issues']:
            if issue['type'] == 'missing_identifier':
                recommendations.append("添加唯一标识符或时间戳列")
            elif issue['type'] == 'missing_expected_columns':
                recommendations.append(f"添加缺少的预期列: {', '.join(issue.get('missing_columns', []))}")
        
        return recommendations
    
    def _calculate_overall_score(self, result: Dict) -> int:
        """计算总体评分
        
        Args:
            result: 验证结果
            
        Returns:
            总体评分(0-100)
        """
        score = 100
        
        # 基于缺失数据扣分
        missing_penalty = min(result['missing_data_percentage'] * 2, 40)
        score -= missing_penalty
        
        # 基于格式问题扣分
        format_penalty = len(result['format_issues']) * 5
        score -= format_penalty
        
        # 基于准确性问题扣分
        accuracy_penalty = 0
        for issue in result['accuracy_issues']:
            if issue.get('severity') == 'high':
                accuracy_penalty += 10
            elif issue.get('severity') == 'medium':
                accuracy_penalty += 5
            else:
                accuracy_penalty += 2
        score -= min(accuracy_penalty, 30)
        
        # 基于结构问题扣分
        structure_penalty = 0
        for issue in result['structure_issues']:
            if issue.get('severity') == 'high':
                structure_penalty += 15
            elif issue.get('severity') == 'medium':
                structure_penalty += 8
            else:
                structure_penalty += 3
        score -= min(structure_penalty, 30)
        
        return max(0, int(score))
    
    def _convert_to_serializable(self, obj):
        """将不可序列化的对象转换为可序列化的形式
        
        Args:
            obj: 需要转换的对象
            
        Returns:
            可序列化的对象
        """
        try:
            # 处理基本类型
            if obj is None or isinstance(obj, (str, int, float)):
                return obj
            elif isinstance(obj, bool):
                return str(obj)
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, pd.Series):
                return self._convert_to_serializable(obj.tolist())
            elif isinstance(obj, pd.DataFrame):
                return self._convert_to_serializable(obj.to_dict())
            elif isinstance(obj, dict):
                return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return self._convert_to_serializable(obj.tolist())
            else:
                # 对于其他类型，尝试转换为字符串
                return str(obj)
        except Exception as e:
            # 如果转换失败，返回字符串表示
            return f"<conversion_error: {str(e)}>"
    
    def generate_report(self, output_path: str = None) -> str:
        """生成验证报告
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            报告文件路径
        """
        if not output_path:
            output_path = self.output_dir / "enhanced_data_validation_report.json"
        
        report = {
            'validation_time': datetime.now().isoformat(),
            'summary': self.summary_stats,
            'detailed_results': self.validation_results,
            'file_classification': self._classify_files(),
            'recommendations': self._generate_overall_recommendations()
        }
        
        # 转换为可序列化的格式
        serializable_report = self._convert_to_serializable(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        return str(output_path)
    
    def _classify_files(self) -> Dict:
        """根据验证结果对文件进行分类
        
        Returns:
            文件分类结果
        """
        classification = {
            'ready_to_use': [],      # 直接可用 (评分 >= 80)
            'needs_preprocessing': [],  # 需要预处理 (评分 >= 60 且 < 80)
            'not_recommended': []   # 不建议使用 (评分 < 60)
        }
        
        for filename, result in self.validation_results.items():
            score = result.get('overall_score', 0)
            if score >= 80:
                classification['ready_to_use'].append({
                    'filename': filename,
                    'score': score,
                    'issues_count': len(result.get('format_issues', [])) + 
                                   len(result.get('accuracy_issues', [])) + 
                                   len(result.get('structure_issues', []))
                })
            elif score >= 60:
                classification['needs_preprocessing'].append({
                    'filename': filename,
                    'score': score,
                    'main_issues': self._get_main_issues(result)
                })
            else:
                classification['not_recommended'].append({
                    'filename': filename,
                    'score': score,
                    'critical_issues': self._get_critical_issues(result)
                })
        
        return classification
    
    def _get_main_issues(self, result: Dict) -> List[str]:
        """获取主要问题
        
        Args:
            result: 验证结果
            
        Returns:
            主要问题列表
        """
        main_issues = []
        
        if result['missing_data_percentage'] > 10:
            main_issues.append(f"高缺失数据率 ({result['missing_data_percentage']}%)")
        
        for issue in result['format_issues']:
            if issue.get('severity') == 'high':
                main_issues.append(f"格式问题: {issue['description']}")
        
        for issue in result['accuracy_issues']:
            if issue.get('severity') == 'high':
                main_issues.append(f"准确性问题: {issue['description']}")
        
        for issue in result['structure_issues']:
            if issue.get('severity') == 'high':
                main_issues.append(f"结构问题: {issue['description']}")
        
        return main_issues[:3]  # 最多返回3个主要问题
    
    def _get_critical_issues(self, result: Dict) -> List[str]:
        """获取关键问题
        
        Args:
            result: 验证结果
            
        Returns:
            关键问题列表
        """
        critical_issues = []
        
        # 安全检查缺失数据百分比
        if isinstance(result.get('missing_data_percentage'), (int, float)) and result['missing_data_percentage'] > 30:
            critical_issues.append(f"极高缺失数据率 ({result['missing_data_percentage']}%)")
        
        # 安全检查结构问题
        structure_issues = result.get('structure_issues', [])
        if isinstance(structure_issues, list):
            for issue in structure_issues:
                if isinstance(issue, dict) and issue.get('severity') == 'high':
                    description = issue.get('description', str(issue))
                    critical_issues.append(f"关键结构问题: {description}")
        
        # 安全检查总体评分
        if isinstance(result.get('overall_score'), (int, float)) and result['overall_score'] < 30:
            critical_issues.append("数据质量极低，不建议使用")
        
        return critical_issues
    
    def _generate_overall_recommendations(self) -> Dict:
        """生成总体建议
        
        Returns:
            总体建议字典
        """
        classification = self._classify_files()
        
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_strategies': []
        }
        
        # 即时行动
        if classification['not_recommended']:
            recommendations['immediate_actions'].append(
                f"优先处理 {len(classification['not_recommended'])} 个不建议使用的文件"
            )
        
        if self.summary_stats['files_with_missing_data'] > 0:
            recommendations['immediate_actions'].append(
                "处理所有缺失数据问题，优先处理高缺失率文件"
            )
        
        # 短期改进
        if classification['needs_preprocessing']:
            recommendations['short_term_improvements'].append(
                f"对 {len(classification['needs_preprocessing'])} 个需要预处理的文件进行标准化处理"
            )
        
        recommendations['short_term_improvements'].extend([
            "统一日期格式和列名规范",
            "建立数据验证检查点",
            "创建数据清洗脚本"
        ])
        
        # 长期策略
        recommendations['long_term_strategies'].extend([
            "建立自动化数据质量监控系统",
            "制定数据收集和存储标准",
            "实施数据版本控制",
            "定期进行数据质量审计"
        ])
        
        return recommendations


def main():
    """主函数"""
    # 设置路径
    data_dir = Path("d:/Mathematical Modeling/2025APMCM/SPEC/2025/data")
    output_dir = Path("d:/Mathematical Modeling/2025APMCM/SPEC/2025")
    
    # 创建验证器
    validator = EnhancedDataValidator(data_dir, output_dir)
    
    # 验证所有文件
    results = validator.validate_all_files()
    
    # 生成报告
    report_path = validator.generate_report()
    
    print(f"\n验证完成! 报告已保存至: {report_path}")
    print(f"总计验证 {validator.summary_stats['total_files']} 个文件")
    print(f"有问题文件: {validator.summary_stats['files_with_issues']} 个")
    print(f"有缺失数据文件: {validator.summary_stats['files_with_missing_data']} 个")
    print(f"有格式问题文件: {validator.summary_stats['files_with_format_issues']} 个")
    print(f"有准确性问题文件: {validator.summary_stats['files_with_accuracy_issues']} 个")
    print(f"有结构问题文件: {validator.summary_stats['files_with_structure_issues']} 个")


if __name__ == "__main__":
    main()
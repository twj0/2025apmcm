#!/usr/bin/env python3
"""
CSV数据完整性检查脚本
用于评估项目中所有CSV文件的数据完整性，包括字段完整性、格式规范性、数值范围等
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CSVDataIntegrityChecker:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.results = {}
        self.summary = {
            'total_files': 0,
            'files_with_issues': 0,
            'issues_by_type': {},
            'recommendations': []
        }
    
    def find_all_csv_files(self):
        """查找所有CSV文件"""
        csv_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        return csv_files
    
    def check_file_integrity(self, file_path):
        """检查单个CSV文件的数据完整性"""
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            # 初始化结果字典
            result = {
                'file_path': file_path,
                'file_name': file_name,
                'file_size_kb': round(os.path.getsize(file_path) / 1024, 2),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'issues': [],
                'warnings': [],
                'missing_data': {},
                'numeric_ranges': {},
                'date_ranges': {},
                'unique_counts': {},
                'duplicate_rows': 0
            }
            
            # 检查基本结构
            if df.empty:
                result['issues'].append("文件为空，没有数据行")
                return result
            
            if len(df.columns) == 0:
                result['issues'].append("文件没有列")
                return result
            
            # 检查缺失值
            missing_counts = df.isnull().sum()
            for col, count in missing_counts.items():
                if count > 0:
                    percentage = round(count / len(df) * 100, 2)
                    result['missing_data'][col] = {
                        'count': int(count),
                        'percentage': percentage
                    }
                    if percentage > 50:
                        result['issues'].append(f"列 '{col}' 缺失值过多 ({percentage}%)")
                    elif percentage > 20:
                        result['warnings'].append(f"列 '{col}' 有较多缺失值 ({percentage}%)")
            
            # 检查重复行
            duplicate_rows = df.duplicated().sum()
            result['duplicate_rows'] = int(duplicate_rows)
            if duplicate_rows > 0:
                duplicate_percentage = round(duplicate_rows / len(df) * 100, 2)
                if duplicate_percentage > 10:
                    result['issues'].append(f"存在过多重复行 ({duplicate_rows} 行, {duplicate_percentage}%)")
                else:
                    result['warnings'].append(f"存在重复行 ({duplicate_rows} 行, {duplicate_percentage}%)")
            
            # 检查数值列的范围和异常值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if not df[col].dropna().empty:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    
                    result['numeric_ranges'][col] = {
                        'min': float(min_val),
                        'max': float(max_val),
                        'mean': float(mean_val),
                        'std': float(std_val)
                    }
                    
                    # 检查可能的异常值 (使用3σ规则)
                    outliers = df[(df[col] < (mean_val - 3*std_val)) | (df[col] > (mean_val + 3*std_val))]
                    if not outliers.empty:
                        outlier_count = len(outliers)
                        outlier_percentage = round(outlier_count / len(df) * 100, 2)
                        result['warnings'].append(f"列 '{col}' 可能有异常值 ({outlier_count} 行, {outlier_percentage}%)")
            
            # 检查日期列
            date_columns = []
            for col in df.columns:
                # 尝试识别日期列
                if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'period']):
                    try:
                        # 尝试转换为日期
                        temp_series = pd.to_datetime(df[col], errors='coerce')
                        if not temp_series.isna().all():
                            date_columns.append(col)
                            min_date = temp_series.min()
                            max_date = temp_series.max()
                            result['date_ranges'][col] = {
                                'min': str(min_date),
                                'max': str(max_date),
                                'span_years': (max_date - min_date).days / 365.25
                            }
                    except:
                        pass
            
            # 检查唯一值计数
            for col in df.columns:
                unique_count = df[col].nunique()
                result['unique_counts'][col] = int(unique_count)
                
                # 如果唯一值很少，可能是分类变量
                if unique_count < 10 and len(df) > 100:
                    result['warnings'].append(f"列 '{col}' 可能是分类变量 (只有 {unique_count} 个唯一值)")
            
            # 检查时间序列数据的连续性
            if date_columns and len(date_columns) == 1:
                date_col = date_columns[0]
                try:
                    date_series = pd.to_datetime(df[date_col], errors='coerce')
                    date_series = date_series.dropna().sort_values()
                    
                    # 检查是否有缺失的时间点
                    if len(date_series) > 1:
                        time_diffs = date_series.diff().dropna()
                        if len(time_diffs.unique()) > 1:
                            result['warnings'].append(f"时间序列 '{date_col}' 可能有不规律的时间间隔")
                except:
                    pass
            
            return result
            
        except Exception as e:
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'error': str(e),
                'issues': [f"无法读取文件: {str(e)}"]
            }
    
    def check_all_files(self):
        """检查所有CSV文件"""
        csv_files = self.find_all_csv_files()
        self.summary['total_files'] = len(csv_files)
        
        for file_path in csv_files:
            print(f"正在检查: {file_path}")
            result = self.check_file_integrity(file_path)
            self.results[os.path.basename(file_path)] = result
            
            # 更新摘要
            if result.get('issues') or result.get('error'):
                self.summary['files_with_issues'] += 1
            
            # 统计问题类型
            for issue in result.get('issues', []):
                issue_type = issue.split(':')[0] if ':' in issue else 'general'
                self.summary['issues_by_type'][issue_type] = self.summary['issues_by_type'].get(issue_type, 0) + 1
        
        # 生成建议
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        # 基于问题类型的建议
        if 'missing_data' in self.summary['issues_by_type']:
            recommendations.append("处理缺失值：考虑使用插值、填充或删除含有过多缺失值的行/列")
        
        if 'duplicate_rows' in self.summary['issues_by_type']:
            recommendations.append("处理重复行：检查并删除重复数据，确保数据唯一性")
        
        if 'outliers' in self.summary['issues_by_type']:
            recommendations.append("处理异常值：检查异常值的合理性，考虑使用稳健统计方法")
        
        if 'time_series' in self.summary['issues_by_type']:
            recommendations.append("时间序列处理：确保时间序列的连续性和规律性")
        
        # 通用建议
        if self.summary['files_with_issues'] > 0:
            recommendations.append("优先处理有严重问题的文件，确保数据质量")
        
        recommendations.append("建立数据质量监控机制，定期检查数据完整性")
        recommendations.append("为所有数据字段添加描述性元数据，提高数据可理解性")
        
        self.summary['recommendations'] = recommendations
    
    def save_report(self, output_path):
        """保存检查报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.summary,
            'detailed_results': self.results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"报告已保存到: {output_path}")
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n=== CSV数据完整性检查摘要 ===")
        print(f"总文件数: {self.summary['total_files']}")
        print(f"有问题的文件数: {self.summary['files_with_issues']}")
        print(f"问题比例: {round(self.summary['files_with_issues'] / self.summary['total_files'] * 100, 2)}%")
        
        print("\n=== 问题类型分布 ===")
        for issue_type, count in self.summary['issues_by_type'].items():
            print(f"{issue_type}: {count}")
        
        print("\n=== 改进建议 ===")
        for i, rec in enumerate(self.summary['recommendations'], 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    # 设置路径
    base_path = "d:\\Mathematical Modeling\\2025APMCM\\SPEC\\2025"
    output_path = "d:\\Mathematical Modeling\\2025APMCM\\SPEC\\2025\\data_integrity_report.json"
    
    # 创建检查器并运行检查
    checker = CSVDataIntegrityChecker(base_path)
    checker.check_all_files()
    
    # 打印摘要并保存报告
    checker.print_summary()
    checker.save_report(output_path)
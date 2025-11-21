#!/usr/bin/env python3
"""
中国海关大豆进口数据获取工具 - 高级分析示例

本脚本展示了如何进行更复杂的数据分析和可视化，包括：
- 趋势分析
- 季节性分析
- 同比环比分析
- 数据可视化
- 预测模型
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher
from ssl_customs_soybean_fetcher import SSLCustomsDataFetcher

class SoybeanDataAnalyzer:
    """大豆进口数据分析器"""
    
    def __init__(self):
        self.fetcher = SimpleCustomsDataFetcher()
        self.data = []
        self.df = None
    
    def collect_data(self, start_year=2022, end_year=2024, months_range=None):
        """收集指定时间范围的数据"""
        print(f"正在收集 {start_year}-{end_year} 年的大豆进口数据...")
        
        if months_range is None:
            months_range = range(1, 13)  # 1-12月
        
        self.data = []
        total_months = (end_year - start_year + 1) * len(months_range)
        processed = 0
        
        for year in range(start_year, end_year + 1):
            for month in months_range:
                processed += 1
                print(f"  进度: {processed}/{total_months} - {year}年{month}月", end="")
                
                data = self.fetcher.get_monthly_import_data(year, month)
                if data:
                    self.data.append(data)
                    print(f" ✓ ({data['total_import']:,.0f}吨)")
                else:
                    print(" ✗ (无数据)")
        
        if self.data:
            self.df = pd.DataFrame(self.data)
            self.df['date'] = pd.to_datetime(self.df['date_str'])
            print(f"\n✓ 成功收集到 {len(self.data)} 个月的数据")
            return True
        else:
            print("\n✗ 未能收集到任何数据")
            return False
    
    def save_data(self, filename="soybean_import_data.csv"):
        """保存数据到文件"""
        if self.df is not None:
            self.df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {filename}")
            return True
        return False
    
    def load_data(self, filename="soybean_import_data.csv"):
        """从文件加载数据"""
        try:
            self.df = pd.read_csv(filename)
            self.df['date'] = pd.to_datetime(self.df['date_str'])
            print(f"从 {filename} 加载了 {len(self.df)} 条数据")
            return True
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False
    
    def basic_statistics(self):
        """基础统计分析"""
        if self.df is None:
            print("没有数据可供分析")
            return
        
        print("\n" + "="*60)
        print("基础统计分析")
        print("="*60)
        
        # 基本统计
        total_import = self.df['total_import'].sum()
        avg_import = self.df['total_import'].mean()
        std_import = self.df['total_import'].std()
        max_import = self.df['total_import'].max()
        min_import = self.df['total_import'].min()
        
        print(f"数据期间: {self.df['date'].min().strftime('%Y年%m月')} - {self.df['date'].max().strftime('%Y年%m月')}")
        print(f"总进口量: {total_import:,.0f} 吨")
        print(f"平均月度进口量: {avg_import:,.0f} 吨")
        print(f"标准差: {std_import:,.0f} 吨")
        print(f"最大月度进口量: {max_import:,.0f} 吨")
        print(f"最小月度进口量: {min_import:,.0f} 吨")
        print(f"变异系数: {(std_import/avg_import)*100:.1f}%")
        
        # 按年份统计
        self.df['year'] = self.df['date'].dt.year
        yearly_stats = self.df.groupby('year')['total_import'].agg(['sum', 'mean', 'count'])
        
        print("\n按年份统计:")
        for year in yearly_stats.index:
            total = yearly_stats.loc[year, 'sum']
            avg = yearly_stats.loc[year, 'mean']
            count = yearly_stats.loc[year, 'count']
            print(f"  {year}年: 总量 {total:,.0f} 吨, 平均 {avg:,.0f} 吨/月, 数据 {count} 个月")
    
    def trend_analysis(self):
        """趋势分析"""
        if self.df is None or len(self.df) < 3:
            print("数据不足，无法进行趋势分析")
            return
        
        print("\n" + "="*60)
        print("趋势分析")
        print("="*60)
        
        # 计算移动平均线
        self.df['ma_3'] = self.df['total_import'].rolling(window=3).mean()
        self.df['ma_6'] = self.df['total_import'].rolling(window=6).mean()
        
        # 计算趋势
        first_half = self.df['total_import'][:len(self.df)//2].mean()
        second_half = self.df['total_import'][len(self.df)//2:].mean()
        
        trend_change = ((second_half - first_half) / first_half) * 100
        
        print(f"整体趋势变化: {trend_change:+.1f}%")
        print(f"前半期平均: {first_half:,.0f} 吨")
        print(f"后半期平均: {second_half:,.0f} 吨")
        
        if trend_change > 5:
            trend_desc = "显著上升"
        elif trend_change < -5:
            trend_desc = "显著下降"
        else:
            trend_desc = "相对稳定"
        
        print(f"趋势描述: {trend_desc}")
    
    def seasonal_analysis(self):
        """季节性分析"""
        if self.df is None:
            print("没有数据可供分析")
            return
        
        print("\n" + "="*60)
        print("季节性分析")
        print("="*60)
        
        self.df['month'] = self.df['date'].dt.month
        monthly_stats = self.df.groupby('month')['total_import'].agg(['mean', 'std', 'count'])
        
        print("月度进口量统计:")
        for month in range(1, 13):
            if month in monthly_stats.index:
                mean_val = monthly_stats.loc[month, 'mean']
                std_val = monthly_stats.loc[month, 'std']
                count = monthly_stats.loc[month, 'count']
                month_name = f"{month}月"
                print(f"  {month_name:>4}: 平均 {mean_val:>8,.0f} 吨 (样本: {count}个月, 标准差: {std_val:,.0f})")
        
        # 找出进口量最高和最低的月份
        if len(monthly_stats) > 0:
            max_month = monthly_stats['mean'].idxmax()
            min_month = monthly_stats['mean'].idxmin()
            
            print(f"\n进口量最高的月份: {max_month}月 (平均 {monthly_stats.loc[max_month, 'mean']:,.0f} 吨)")
            print(f"进口量最低的月份: {min_month}月 (平均 {monthly_stats.loc[min_month, 'mean']:,.0f} 吨)")
    
    def year_over_year_analysis(self):
        """同比分析"""
        if self.df is None:
            print("没有数据可供分析")
            return
        
        print("\n" + "="*60)
        print("同比分析")
        print("="*60)
        
        # 确保数据按时间排序
        self.df = self.df.sort_values('date')
        
        # 计算同比变化
        yoy_changes = []
        for i in range(12, len(self.df)):
            current_month = self.df.iloc[i]['total_import']
            same_month_last_year = self.df.iloc[i-12]['total_import']
            
            if same_month_last_year > 0:
                yoy_change = ((current_month - same_month_last_year) / same_month_last_year) * 100
                yoy_changes.append({
                    'date': self.df.iloc[i]['date'],
                    'yoy_change': yoy_change,
                    'current': current_month,
                    'last_year': same_month_last_year
                })
        
        if yoy_changes:
            yoy_df = pd.DataFrame(yoy_changes)
            
            print(f"同比分析结果 (共 {len(yoy_changes)} 个对比月份):")
            print(f"平均同比变化: {yoy_df['yoy_change'].mean():+.1f}%")
            print(f"同比变化标准差: {yoy_df['yoy_change'].std():.1f}%")
            print(f"最大同比增长: {yoy_df['yoy_change'].max():+.1f}%")
            print(f"最大同比下降: {yoy_df['yoy_change'].min():+.1f}%")
            
            # 显示最近几个同比数据
            print("\n最近几个月的同比变化:")
            for _, row in yoy_df.tail(5).iterrows():
                print(f"  {row['date'].strftime('%Y年%m月')}: {row['yoy_change']:+6.1f}% "
                      f"({row['current']:,.0f} 吨 vs {row['last_year']:,.0f} 吨)")
    
    def create_visualizations(self):
        """创建数据可视化"""
        if self.df is None:
            print("没有数据可供可视化")
            return
        
        print("\n" + "="*60)
        print("创建数据可视化")
        print("="*60)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 时间序列图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(self.df['date'], self.df['total_import'], marker='o', linewidth=2)
        plt.title('大豆月度进口量时间序列')
        plt.xlabel('时间')
        plt.ylabel('进口量 (吨)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 月度箱线图
        plt.subplot(2, 2, 2)
        monthly_data = [self.df[self.df['month'] == m]['total_import'].values 
                       for m in range(1, 13) if m in self.df['month'].values]
        plt.boxplot(monthly_data, labels=[f'{m}月' for m in range(1, len(monthly_data)+1)])
        plt.title('月度进口量分布')
        plt.xlabel('月份')
        plt.ylabel('进口量 (吨)')
        plt.xticks(rotation=45)
        
        # 3. 年度对比
        plt.subplot(2, 2, 3)
        yearly_data = self.df.groupby('year')['total_import'].sum()
        plt.bar(yearly_data.index, yearly_data.values)
        plt.title('年度总进口量对比')
        plt.xlabel('年份')
        plt.ylabel('总进口量 (吨)')
        
        # 4. 移动平均
        plt.subplot(2, 2, 4)
        plt.plot(self.df['date'], self.df['total_import'], 'o-', alpha=0.6, label='原始数据')
        if 'ma_3' in self.df.columns:
            plt.plot(self.df['date'], self.df['ma_3'], '-', linewidth=2, label='3个月移动平均')
        if 'ma_6' in self.df.columns:
            plt.plot(self.df['date'], self.df['ma_6'], '-', linewidth=2, label='6个月移动平均')
        plt.title('移动平均趋势')
        plt.xlabel('时间')
        plt.ylabel('进口量 (吨)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('soybean_analysis_charts.png', dpi=300, bbox_inches='tight')
        print("可视化图表已保存到: soybean_analysis_charts.png")
        
        # 显示图表（如果在支持的环境中）
        try:
            plt.show()
        except:
            print("无法显示图表，图表已保存到文件")
    
    def generate_report(self):
        """生成分析报告"""
        if self.df is None:
            print("没有数据可供生成报告")
            return
        
        print("\n" + "="*60)
        print("生成分析报告")
        print("="*60)
        
        report = []
        report.append("中国大豆进口数据分析报告")
        report.append("=" * 40)
        report.append(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        report.append(f"数据期间: {self.df['date'].min().strftime('%Y年%m月')} - {self.df['date'].max().strftime('%Y年%m月')}")
        report.append(f"数据点数量: {len(self.df)} 个月")
        report.append("")
        
        # 基础统计
        total_import = self.df['total_import'].sum()
        avg_import = self.df['total_import'].mean()
        
        report.append("一、基础统计")
        report.append(f"  总进口量: {total_import:,.0f} 吨")
        report.append(f"  平均月度进口量: {avg_import:,.0f} 吨")
        report.append(f"  变异系数: {(self.df['total_import'].std()/avg_import)*100:.1f}%")
        report.append("")
        
        # 趋势分析
        if len(self.df) >= 6:
            first_half = self.df['total_import'][:len(self.df)//2].mean()
            second_half = self.df['total_import'][len(self.df)//2:].mean()
            trend_change = ((second_half - first_half) / first_half) * 100
            
            report.append("二、趋势分析")
            report.append(f"  整体趋势变化: {trend_change:+.1f}%")
            report.append(f"  趋势描述: {'显著上升' if trend_change > 5 else '显著下降' if trend_change < -5 else '相对稳定'}")
            report.append("")
        
        # 季节性分析
        self.df['month'] = self.df['date'].dt.month
        monthly_stats = self.df.groupby('month')['total_import'].mean()
        
        if len(monthly_stats) > 0:
            max_month = monthly_stats.idxmax()
            min_month = monthly_stats.idxmin()
            
            report.append("三、季节性分析")
            report.append(f"  进口量最高的月份: {max_month}月 (平均 {monthly_stats[max_month]:,.0f} 吨)")
            report.append(f"  进口量最低的月份: {min_month}月 (平均 {monthly_stats[min_month]:,.0f} 吨)")
            report.append("")
        
        # 结论
        report.append("四、结论与建议")
        report.append("  1. 继续监控月度进口量变化趋势")
        report.append("  2. 关注季节性波动规律")
        report.append("  3. 结合国际市场情况进行综合分析")
        report.append("")
        
        report_content = "\n".join(report)
        
        # 保存报告
        with open('soybean_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("分析报告已保存到: soybean_analysis_report.txt")
        print("\n报告预览:")
        print(report_content)

def main():
    """主函数"""
    print("中国海关大豆进口数据 - 高级分析示例")
    print("=" * 60)
    print()
    
    # 创建分析器
    analyzer = SoybeanDataAnalyzer()
    
    # 选项菜单
    options = [
        ("1", "收集数据", analyzer.collect_data),
        ("2", "加载已有数据", analyzer.load_data),
        ("3", "基础统计分析", analyzer.basic_statistics),
        ("4", "趋势分析", analyzer.trend_analysis),
        ("5", "季节性分析", analyzer.seasonal_analysis),
        ("6", "同比分析", analyzer.year_over_year_analysis),
        ("7", "创建可视化", analyzer.create_visualizations),
        ("8", "生成分析报告", analyzer.generate_report),
        ("9", "保存数据", analyzer.save_data),
        ("0", "退出", None)
    ]
    
    while True:
        print("\n功能菜单:")
        for key, desc, _ in options:
            print(f"  {key}. {desc}")
        
        choice = input("\n请选择功能 (0-9): ").strip()
        
        if choice == "0":
            print("感谢使用，再见!")
            break
        
        # 找到对应的函数
        selected_option = None
        for key, desc, func in options:
            if key == choice:
                selected_option = (key, desc, func)
                break
        
        if selected_option and selected_option[2]:
            try:
                # 对于需要参数的特殊处理
                if selected_option[0] in ["1"]:
                    # 数据收集需要参数
                    start_year = int(input("开始年份 (如 2022): ") or "2022")
                    end_year = int(input("结束年份 (如 2024): ") or "2024")
                    selected_option[2](start_year, end_year)
                else:
                    selected_option[2]()
            except Exception as e:
                print(f"执行出错: {e}")
                print("请重试或选择其他功能")
        else:
            print("无效选择，请重试")

if __name__ == "__main__":
    main()
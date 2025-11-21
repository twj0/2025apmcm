#!/usr/bin/env python3
"""
中国海关大豆进口数据获取工具测试脚本

本脚本用于测试simple_customs_soybean_fetcher.py的功能。

作者: APMCM 2025C 团队
日期: 2025-01-20
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# 添加脚本路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher


class TestSimpleCustomsDataFetcher(unittest.TestCase):
    """测试SimpleCustomsDataFetcher类"""
    
    def setUp(self):
        """测试前的设置"""
        self.fetcher = SimpleCustomsDataFetcher()
    
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.fetcher.soybean_hs_codes, dict)
        self.assertIn('种用大豆', self.fetcher.soybean_hs_codes)
        self.assertIn('黄大豆', self.fetcher.soybean_hs_codes)
        self.assertEqual(self.fetcher.soybean_hs_codes['种用大豆'], '1201100000')
    
    def test_format_monthly_data(self):
        """测试月度数据格式化"""
        data = {'quantity': 1000000}
        result = self.fetcher._format_monthly_data(2024, 1, data, 'test')
        
        self.assertEqual(result['year'], 2024)
        self.assertEqual(result['month'], 1)
        self.assertEqual(result['date_str'], '2024-01')
        self.assertEqual(result['total_import'], 1000000)
        self.assertEqual(result['source'], 'test')
        self.assertIn('query_time', result)
    
    def test_extract_date_from_text(self):
        """测试从文本中提取日期"""
        # 测试不同格式的日期
        test_cases = [
            ('2024年1月1日', '2024-01-01'),
            ('2024年1月', '2024-01'),
            ('2024-01-01', '2024-01-01'),
            ('2024/1/1', '2024-01-01'),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.fetcher._extract_date_from_text(text)
                self.assertEqual(result, expected)
    
    def test_extract_quantity_from_html(self):
        """测试从HTML中提取数量"""
        from bs4 import BeautifulSoup
        
        # 测试HTML内容
        html_content = """
        <html>
            <body>
                <p>本月大豆进口量为123.45万吨</p>
            </body>
        </html>
        """
        
        soup = BeautifulSoup(html_content, 'html.parser')
        result = self.fetcher._extract_quantity_from_html(soup)
        
        # 123.45万吨 = 1234500吨
        self.assertEqual(result, 1234500.0)
    
    def test_extract_soybean_data_from_content(self):
        """测试从公告内容中提取大豆数据"""
        content = "本月大豆进口量为123.45万吨，同比增长5%"
        result = self.fetcher._extract_soybean_data_from_content(content)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['quantity'], 1234500.0)
        self.assertEqual(result['unit'], '吨')
        self.assertEqual(result['original_quantity'], 123.45)
        self.assertEqual(result['original_unit'], '万吨')
    
    def test_save_data_to_csv(self):
        """测试保存数据到CSV"""
        data = {
            'year': 2024,
            'month': 1,
            'date_str': '2024-01',
            'total_import': 1000000,
            'source': 'test',
            'query_time': '2025-01-20 12:00:00'
        }
        
        filename = 'test_soybean_data.csv'
        result = self.fetcher.save_data_to_csv(data, filename)
        
        self.assertTrue(result)
        
        # 验证文件是否存在
        filepath = f"../data/raw/{filename}"
        self.assertTrue(os.path.exists(filepath))
        
        # 验证文件内容
        df = pd.read_csv(filepath)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['year'], 2024)
        self.assertEqual(df.iloc[0]['month'], 1)
        self.assertEqual(df.iloc[0]['total_import'], 1000000)
        
        # 清理测试文件
        os.remove(filepath)
    
    @patch('requests.Session.get')
    def test_get_monthly_import_data_with_mock(self, mock_get):
        """使用模拟测试获取月度进口数据"""
        # 模拟响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.json.return_value = {
            'success': True,
            'data': {'quantity': 1000000}
        }
        mock_get.return_value = mock_response
        
        # 测试获取数据
        result = self.fetcher.get_monthly_import_data(2024, 1)
        
        # 验证结果
        self.assertIsNotNone(result)
        self.assertEqual(result['year'], 2024)
        self.assertEqual(result['month'], 1)
        self.assertEqual(result['total_import'], 1000000)


def run_integration_test():
    """运行集成测试"""
    print("运行集成测试...")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = SimpleCustomsDataFetcher()
    
    # 测试获取2024年1月的数据
    print("测试获取2024年1月的大豆进口数据...")
    data = fetcher.get_monthly_import_data(2024, 1)
    
    if data:
        print("✓ 数据获取成功!")
        print(f"  年份: {data['year']}")
        print(f"  月份: {data['month']}")
        print(f"  进口量: {data['total_import']} 吨")
        print(f"  数据来源: {data['source']}")
        
        # 测试保存数据
        print("\n测试保存数据...")
        filename = f"test_soybean_import_{data['year']}_{data['month']:02d}.csv"
        success = fetcher.save_data_to_csv(data, filename)
        
        if success:
            print(f"✓ 数据已成功保存到 {filename}")
            
            # 验证文件内容
            filepath = f"../data/raw/{filename}"
            df = pd.read_csv(filepath)
            print(f"✓ 文件验证成功，包含 {len(df)} 行数据")
            
            # 清理测试文件
            os.remove(filepath)
            print("✓ 测试文件已清理")
        else:
            print("✗ 数据保存失败")
    else:
        print("✗ 数据获取失败")
        print("  这可能是由于:")
        print("  1. 网络连接问题")
        print("  2. 海关网站结构变化")
        print("  3. 数据暂时不可用")
        print("  4. 需要更复杂的认证或参数")
    
    print("=" * 50)
    print("集成测试完成")


def main():
    """主函数"""
    print("中国海关大豆进口数据获取工具测试")
    print("=" * 50)
    
    # 运行单元测试
    print("运行单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 50)
    
    # 运行集成测试
    run_integration_test()


if __name__ == "__main__":
    main()
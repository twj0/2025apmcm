#!/usr/bin/env python3
"""
简单测试脚本 - 测试海关数据获取工具的基本功能
"""

import unittest
import sys
import os
from datetime import datetime

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher
    from ssl_customs_soybean_fetcher import SSLCustomsDataFetcher
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保脚本文件存在于同一目录中")
    sys.exit(1)

class TestCustomsDataFetcher(unittest.TestCase):
    """测试海关数据获取工具"""
    
    def setUp(self):
        """设置测试环境"""
        self.simple_fetcher = SimpleCustomsDataFetcher()
        self.ssl_fetcher = SSLCustomsDataFetcher()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.simple_fetcher.session)
        self.assertIsNotNone(self.simple_fetcher.soybean_hs_codes)
        self.assertIn('大豆(总计)', self.simple_fetcher.soybean_hs_codes)
        
        self.assertIsNotNone(self.ssl_fetcher.session)
        self.assertIsNotNone(self.ssl_fetcher.soybean_hs_codes)
        self.assertIn('大豆(总计)', self.ssl_fetcher.soybean_hs_codes)
    
    def test_date_extraction(self):
        """测试日期提取功能"""
        test_cases = [
            ("2024年1月", "2024-01"),
            ("2024年01月", "2024-01"),
            ("2024年1月15日", "2024-01-15"),
            ("2024/1/15", "2024-01-15"),
            ("2024-01-15", "2024-01-15"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.simple_fetcher._extract_date_from_text(input_text)
                self.assertEqual(result, expected)
    
    def test_quantity_extraction(self):
        """测试数量提取功能"""
        from bs4 import BeautifulSoup
        
        test_cases = [
            ("进口大豆100万吨", 1000000),
            ("大豆进口50吨", 50),
            ("进口大豆2000千克", 2),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                soup = BeautifulSoup(f"<div>{text}</div>", 'html.parser')
                result = self.simple_fetcher._extract_quantity_from_html(soup)
                self.assertEqual(result, expected)
    
    def test_soybean_data_extraction(self):
        """测试从文本中提取大豆数据"""
        test_cases = [
            ("本月大豆进口100万吨", {
                'quantity': 1000000,
                'unit': '吨',
                'original_quantity': 100,
                'original_unit': '万吨',
                'extracted_from': '公告内容'
            }),
            ("进口大豆50吨", {
                'quantity': 50,
                'unit': '吨',
                'original_quantity': 50,
                'original_unit': '吨',
                'extracted_from': '公告内容'
            }),
        ]
        
        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.simple_fetcher._extract_soybean_data_from_content(text)
                self.assertEqual(result, expected)
    
    def test_data_formatting(self):
        """测试数据格式化"""
        raw_data = {'quantity': 1000000}
        result = self.simple_fetcher._format_monthly_data(2024, 1, raw_data, 'test')
        
        self.assertEqual(result['year'], 2024)
        self.assertEqual(result['month'], 1)
        self.assertEqual(result['date_str'], '2024-01')
        self.assertEqual(result['total_import'], 1000000)
        self.assertEqual(result['source'], 'test')
        self.assertIsNotNone(result['query_time'])
    
    def test_csv_saving(self):
        """测试CSV保存功能"""
        test_data = {
            'year': 2024,
            'month': 1,
            'date_str': '2024-01',
            'total_import': 1000000,
            'source': 'test',
            'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        filename = 'test_soybean_import.csv'
        result = self.simple_fetcher.save_data_to_csv(test_data, filename)
        
        self.assertTrue(result)
        
        # 检查文件是否创建
        filepath = f"../data/raw/{filename}"
        self.assertTrue(os.path.exists(filepath))
        
        # 清理测试文件
        if os.path.exists(filepath):
            os.remove(filepath)
    
    @unittest.skip("需要网络连接，仅在有需要时运行")
    def test_network_connection(self):
        """测试网络连接 - 需要网络连接"""
        # 这个测试需要网络连接，默认跳过
        # 可以在需要时手动启用
        try:
            response = self.simple_fetcher.session.get(
                "https://www.customs.gov.cn", 
                timeout=10
            )
            self.assertEqual(response.status_code, 200)
        except Exception as e:
            self.fail(f"网络连接测试失败: {e}")
    
    @unittest.skip("需要网络连接，仅在有需要时运行")
    def test_ssl_network_connection(self):
        """测试SSL网络连接 - 需要网络连接"""
        # 这个测试需要网络连接，默认跳过
        # 可以在需要时手动启用
        try:
            response = self.ssl_fetcher.session.get(
                "https://www.customs.gov.cn", 
                timeout=10,
                verify=False
            )
            self.assertEqual(response.status_code, 200)
        except Exception as e:
            self.fail(f"SSL网络连接测试失败: {e}")


def run_basic_functionality_test():
    """运行基本功能测试"""
    print("=" * 60)
    print("运行基本功能测试")
    print("=" * 60)
    
    # 创建获取器实例
    print("1. 创建数据获取器实例...")
    try:
        fetcher = SimpleCustomsDataFetcher()
        ssl_fetcher = SSLCustomsDataFetcher()
        print("   ✓ 简单版获取器创建成功")
        print("   ✓ SSL版获取器创建成功")
    except Exception as e:
        print(f"   ✗ 创建失败: {e}")
        return False
    
    # 测试日期提取
    print("\n2. 测试日期提取功能...")
    test_dates = [
        ("2024年1月", "2024-01"),
        ("2024年01月", "2024-01"),
        ("2024年1月15日", "2024-01-15"),
    ]
    
    for input_text, expected in test_dates:
        result = fetcher._extract_date_from_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"   {status} '{input_text}' -> '{result}' (期望: '{expected}')")
    
    # 测试数量提取
    print("\n3. 测试数量提取功能...")
    from bs4 import BeautifulSoup
    
    test_quantities = [
        ("进口大豆100万吨", 1000000),
        ("大豆进口50吨", 50),
        ("进口大豆2000千克", 2),
    ]
    
    for text, expected in test_quantities:
        soup = BeautifulSoup(f"<div>{text}</div>", 'html.parser')
        result = fetcher._extract_quantity_from_html(soup)
        status = "✓" if result == expected else "✗"
        print(f"   {status} '{text}' -> {result} (期望: {expected})")
    
    # 测试数据格式化
    print("\n4. 测试数据格式化功能...")
    raw_data = {'quantity': 1000000}
    formatted = fetcher._format_monthly_data(2024, 1, raw_data, 'test')
    
    checks = [
        (formatted['year'], 2024, "年份"),
        (formatted['month'], 1, "月份"),
        (formatted['date_str'], "2024-01", "日期字符串"),
        (formatted['total_import'], 1000000, "总进口量"),
        (formatted['source'], "test", "数据源"),
    ]
    
    for actual, expected, name in checks:
        status = "✓" if actual == expected else "✗"
        print(f"   {status} {name}: {actual} (期望: {expected})")
    
    # 测试CSV保存
    print("\n5. 测试CSV保存功能...")
    test_data = {
        'year': 2024,
        'month': 1,
        'date_str': '2024-01',
        'total_import': 1000000,
        'source': 'test',
        'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    filename = 'test_soybean_import.csv'
    save_result = fetcher.save_data_to_csv(test_data, filename)
    filepath = f"../data/raw/{filename}"
    
    if save_result and os.path.exists(filepath):
        print("   ✓ CSV保存成功")
        # 清理测试文件
        os.remove(filepath)
        print("   ✓ 测试文件已清理")
    else:
        print("   ✗ CSV保存失败")
    
    print("\n基本功能测试完成!")
    return True


if __name__ == "__main__":
    print("海关数据获取工具测试脚本")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        # 运行基本功能测试
        run_basic_functionality_test()
    else:
        # 运行单元测试
        print("运行单元测试...")
        unittest.main(verbosity=2)
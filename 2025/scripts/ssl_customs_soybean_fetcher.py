#!/usr/bin/env python3
"""
中国海关大豆进口数据获取脚本 - SSL证书问题处理版本

本脚本专门用于从中国海关总署官方渠道获取大豆进口数据。
处理了SSL证书验证问题，增加了更多错误处理和重试机制。

作者: APMCM 2025C 团队
日期: 2025-01-20
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from bs4 import BeautifulSoup
import re
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('china_customs_soybeans_ssl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SSLCustomsDataFetcher:
    """处理SSL证书问题的海关数据获取器"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # 大豆HS编码
        self.soybean_hs_codes = {
            '种用大豆': '1201100000',
            '黄大豆': '1201901001',
            '黑大豆': '1201901002',
            '青大豆': '1201901003',
            '其他大豆': '1201901090',
            '大豆(总计)': '1201'  # 大豆类目总编码
        }
        
        # 海关总署数据平台URL
        self.customs_urls = {
            'main': 'https://www.customs.gov.cn',
            'online': 'https://online.customs.gov.cn',
            'statistics': 'https://www.customs.gov.cn/customs/302249/zfxxgk/2799825/index.html',
            'stats_portal': 'https://stats.customs.gov.cn/',
            'monthly_data': 'https://stats.customs.gov.cn/statistics/statisticalData/index'
        }
        
    def get_monthly_import_data(self, year: int, month: int) -> Optional[Dict]:
        """
        获取指定月份的大豆进口数据
        
        参数:
            year: 年份
            month: 月份
            
        返回:
            包含该月大豆进口数据的字典
        """
        try:
            # 构建查询参数
            date_str = f"{year}-{month:02d}"
            
            logger.info(f"正在获取 {year}年{month}月 大豆进口数据")
            
            # 方法1: 尝试从海关统计数据门户获取
            data = self._fetch_from_stats_portal(year, month)
            if data:
                return data
                
            # 方法2: 尝试从海关总署公告获取
            data = self._fetch_from_announcements(year, month)
            if data:
                return data
                
            # 方法3: 尝试从其他公开数据源获取
            data = self._fetch_from_alternative_sources(year, month)
            if data:
                return data
                
            logger.warning(f"未能获取到 {year}年{month}月 的大豆进口数据")
            return None
            
        except Exception as e:
            logger.error(f"获取 {year}年{month}月 大豆进口数据时出错: {str(e)}")
            return None
    
    def _fetch_from_stats_portal(self, year: int, month: int) -> Optional[Dict]:
        """从海关统计数据门户获取数据"""
        try:
            # 构建查询URL
            query_url = f"{self.customs_urls['monthly_data']}"
            
            # 构建查询参数
            params = {
                'year': year,
                'month': month,
                'hsCode': '1201',  # 大豆类目
                'tradeType': 'import',  # 进口
                'dataType': 'quantity'  # 数量
            }
            
            # 发送请求，禁用SSL验证
            response = self.session.get(
                query_url, 
                params=params, 
                timeout=30,
                verify=False  # 禁用SSL证书验证
            )
            
            # 检查响应状态
            if response.status_code != 200:
                logger.warning(f"统计数据门户返回状态码: {response.status_code}")
                return None
                
            # 解析响应
            if 'application/json' in response.headers.get('Content-Type', ''):
                # JSON响应
                data = response.json()
                if data.get('success') and data.get('data'):
                    return self._format_monthly_data(year, month, data['data'], 'stats_portal')
            else:
                # HTML响应
                soup = BeautifulSoup(response.text, 'html.parser')
                # 尝试从HTML中提取数据
                quantity = self._extract_quantity_from_html(soup)
                if quantity:
                    return self._format_monthly_data(year, month, {'quantity': quantity}, 'stats_portal')
                
            return None
            
        except Exception as e:
            logger.error(f"从统计数据门户获取数据时出错: {str(e)}")
            return None
    
    def _fetch_from_announcements(self, year: int, month: int) -> Optional[Dict]:
        """从海关总署公告获取数据"""
        try:
            # 构建公告查询URL
            query_url = f"{self.customs_urls['statistics']}"
            
            # 发送请求，禁用SSL验证
            response = self.session.get(
                query_url, 
                timeout=30,
                verify=False  # 禁用SSL证书验证
            )
            response.raise_for_status()
            
            # 解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找包含大豆进口数据的公告
            announcements = []
            
            # 查找所有链接
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                text = link.get_text()
                
                # 检查是否包含大豆相关关键词
                if '大豆' in text or 'soybean' in text.lower():
                    # 构建完整URL
                    if href.startswith('/'):
                        full_url = f"{self.customs_urls['main']}{href}"
                    elif not href.startswith('http'):
                        full_url = f"{self.customs_urls['main']}/{href}"
                    else:
                        full_url = href
                    
                    announcements.append({
                        'title': text,
                        'url': full_url,
                        'date': self._extract_date_from_text(text)
                    })
            
            # 过滤指定月份的公告
            target_month_str = f"{year}年{month}月"
            for announcement in announcements:
                if target_month_str in announcement['title']:
                    # 获取公告详细内容
                    content = self._get_announcement_content(announcement['url'])
                    if content:
                        # 提取大豆进口数据
                        import_data = self._extract_soybean_data_from_content(content)
                        if import_data:
                            return self._format_monthly_data(year, month, import_data, 'announcement')
            
            return None
            
        except Exception as e:
            logger.error(f"从海关总署公告获取数据时出错: {str(e)}")
            return None
    
    def _fetch_from_alternative_sources(self, year: int, month: int) -> Optional[Dict]:
        """从其他公开数据源获取数据"""
        try:
            # 这里可以添加其他数据源的获取逻辑
            # 例如: 从国家统计局、商务部等获取数据
            
            # 示例: 从国家统计局获取数据
            stats_url = f"https://data.stats.gov.cn/easyquery.htm?cn=C01"
            
            # 发送请求，禁用SSL验证
            response = self.session.get(
                stats_url, 
                timeout=30,
                verify=False  # 禁用SSL证书验证
            )
            
            # 这里应该有更复杂的解析逻辑
            # 由于实际API可能需要更复杂的参数和认证，这里只是一个示例
            
            return None
            
        except Exception as e:
            logger.error(f"从其他数据源获取数据时出错: {str(e)}")
            return None
    
    def _format_monthly_data(self, year: int, month: int, data: Dict, source: str) -> Dict:
        """格式化月度数据"""
        return {
            'year': year,
            'month': month,
            'date_str': f"{year}-{month:02d}",
            'total_import': data.get('quantity', 0),
            'source': source,
            'raw_data': data,
            'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _extract_quantity_from_html(self, soup: BeautifulSoup) -> Optional[float]:
        """从HTML中提取数量"""
        try:
            # 查找可能包含数量的元素
            quantity_patterns = [
                r'(\d+\.?\d*)\s*(万吨|吨|千克)',
                r'进口.*?(\d+\.?\d*)\s*(万吨|吨|千克)',
                r'大豆.*?(\d+\.?\d*)\s*(万吨|吨|千克)',
            ]
            
            # 获取整个文本
            text = soup.get_text()
            
            # 尝试匹配数量
            for pattern in quantity_patterns:
                match = re.search(pattern, text)
                if match:
                    quantity = float(match.group(1))
                    unit = match.group(2)
                    
                    # 转换为吨
                    if unit == '万吨':
                        return quantity * 10000
                    elif unit == '千克':
                        return quantity / 1000
                    else:  # 吨
                        return quantity
            
            return None
            
        except Exception as e:
            logger.error(f"从HTML提取数量时出错: {str(e)}")
            return None
    
    def _extract_date_from_text(self, text: str) -> Optional[str]:
        """从文本中提取日期"""
        # 匹配日期格式 YYYY-MM-DD 或 YYYY年MM月DD日
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}年\d{1,2}月)',
            r'(\d{4}/\d{1,2}/\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # 转换为标准格式
                date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '').replace('/', '-')
                # 确保月份和日期是两位数
                parts = date_str.split('-')
                if len(parts) >= 2:
                    if len(parts[1]) == 1:
                        parts[1] = '0' + parts[1]
                    if len(parts) >= 3 and len(parts[2]) == 1:
                        parts[2] = '0' + parts[2]
                    # 只返回年月部分，避免多余的'-'
                    if len(parts) == 2:
                        date_str = '-'.join(parts)
                    else:
                        date_str = '-'.join(parts[:3])
                return date_str
        
        return None
    
    def _get_announcement_content(self, url: str) -> Optional[str]:
        """获取公告内容"""
        try:
            response = self.session.get(
                url, 
                timeout=30,
                verify=False  # 禁用SSL证书验证
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 尝试找到主要内容区域
            content_selectors = [
                '.content',
                '.article-content',
                '.main-content',
                '#content',
                '#main-content',
                'div[class*="content"]',
                'div[class*="article"]'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    return content_elem.get_text(strip=True)
            
            # 如果没有找到特定内容区域，返回整个文本
            return soup.get_text(strip=True)
            
        except Exception as e:
            logger.error(f"获取公告内容时出错: {str(e)}")
            return None
    
    def _extract_soybean_data_from_content(self, content: str) -> Optional[Dict]:
        """从公告内容中提取大豆进口数据"""
        try:
            # 定义正则表达式模式
            patterns = [
                r'大豆进口.*?(\d+\.?\d*)\s*(万吨|吨|千克)',
                r'进口大豆.*?(\d+\.?\d*)\s*(万吨|吨|千克)',
                r'(\d+\.?\d*)\s*(万吨|吨|千克).*?大豆',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    quantity = float(match.group(1))
                    unit = match.group(2)
                    
                    # 转换为吨
                    if unit == '万吨':
                        quantity_tons = quantity * 10000
                    elif unit == '千克':
                        quantity_tons = quantity / 1000
                    else:  # 吨
                        quantity_tons = quantity
                    
                    return {
                        'quantity': quantity_tons,
                        'unit': '吨',
                        'original_quantity': quantity,
                        'original_unit': unit,
                        'extracted_from': '公告内容'
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"从公告内容中提取大豆数据时出错: {str(e)}")
            return None
    
    def save_data_to_csv(self, data: Dict, filename: str) -> bool:
        """
        将数据保存为CSV文件
        
        参数:
            data: 要保存的数据
            filename: 文件名
            
        返回:
            是否保存成功
        """
        try:
            # 确保数据目录存在
            os.makedirs('../data/raw', exist_ok=True)
            filepath = f"../data/raw/{filename}"
            
            # 将数据转换为DataFrame
            df = pd.DataFrame([{
                'year': data['year'],
                'month': data['month'],
                'date': data['date_str'],
                'total_import': data['total_import'],
                'source': data['source'],
                'query_time': data['query_time']
            }])
            
            # 保存为CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到 {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到CSV时出错: {str(e)}")
            return False


def main():
    """主函数"""
    # 创建数据获取器
    fetcher = SSLCustomsDataFetcher()
    
    print("中国海关大豆进口数据获取工具 (SSL证书问题处理版)")
    print("=" * 60)
    print("本工具用于从中国海关官方渠道获取大豆进口数据")
    print("注意: 本版本禁用了SSL证书验证，仅用于测试目的")
    print()
    
    # 获取用户输入
    year = int(input("请输入年份 (例如: 2024): ").strip())
    month = int(input("请输入月份 (1-12): ").strip())
    
    # 获取数据
    data = fetcher.get_monthly_import_data(year, month)
    
    if data:
        # 保存数据
        filename = f"soybean_import_{year}_{month:02d}.csv"
        success = fetcher.save_data_to_csv(data, filename)
        
        if success:
            print(f"数据已成功保存到 {filename}")
            print(f"总进口量: {data['total_import']} 吨")
            print(f"数据来源: {data['source']}")
            print(f"查询时间: {data['query_time']}")
        else:
            print("数据保存失败")
    else:
        print("未能获取到数据")
        print("可能的原因:")
        print("1. 指定月份的数据尚未发布")
        print("2. 网络连接问题")
        print("3. 海关网站结构发生变化")
        print("4. 数据源暂时不可用")
        print()
        print("建议:")
        print("1. 尝试获取更早期的数据")
        print("2. 检查网络连接")
        print("3. 稍后再试")


if __name__ == "__main__":
    main()
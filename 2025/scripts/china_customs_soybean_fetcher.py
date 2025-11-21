#!/usr/bin/env python3
"""
中国海关大豆进口数据获取脚本

本脚本用于从中国海关官方渠道获取大豆进口数据。
支持多种数据获取方式：
1. 通过海关总署在线查询平台获取数据
2. 通过海关统计数据API获取数据
3. 通过爬虫方式获取公开数据

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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('china_customs_soybeans.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChinaCustomsDataFetcher:
    """中国海关数据获取器"""
    
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
            '其他大豆': '1201901090'
        }
        
        # 海关总署数据平台URL
        self.customs_urls = {
            'main': 'https://www.customs.gov.cn',
            'online': 'https://online.customs.gov.cn',
            'statistics': 'https://www.customs.gov.cn/customs/302249/zfxxgk/2799825/index.html',
            'data_query': 'https://stats.customs.gov.cn/'
        }
        
    def get_soybean_import_data_by_hs_code(self, hs_code: str, start_date: str, end_date: str) -> Optional[Dict]:
        """
        根据HS编码获取大豆进口数据
        
        参数:
            hs_code: 大豆HS编码
            start_date: 开始日期，格式YYYY-MM
            end_date: 结束日期，格式YYYY-MM
            
        返回:
            包含进口数据的字典
        """
        try:
            # 构建查询URL
            query_url = f"{self.customs_urls['data_query']}/moduledata/index"
            
            # 构建查询参数
            params = {
                'hsCode': hs_code,
                'startDate': start_date,
                'endDate': end_date,
                'tradeType': 'import',  # 进口
                'dataType': 'quantity'  # 数量
            }
            
            logger.info(f"正在查询HS编码 {hs_code} 的大豆进口数据，时间范围: {start_date} 至 {end_date}")
            
            # 发送请求
            response = self.session.get(query_url, params=params, timeout=30)
            response.raise_for_status()
            
            # 解析响应数据
            data = response.json()
            
            # 提取大豆进口数据
            if data.get('success') and data.get('data'):
                soybean_data = {
                    'hs_code': hs_code,
                    'product_name': self._get_product_name_by_hs_code(hs_code),
                    'import_data': data['data'],
                    'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'start_date': start_date,
                    'end_date': end_date
                }
                return soybean_data
            else:
                logger.warning(f"未找到HS编码 {hs_code} 的数据")
                return None
                
        except Exception as e:
            logger.error(f"获取HS编码 {hs_code} 数据时出错: {str(e)}")
            return None
    
    def get_soybean_import_data_by_month(self, year: int, month: int) -> Optional[Dict]:
        """
        获取指定月份的大豆进口数据
        
        参数:
            year: 年份
            month: 月份
            
        返回:
            包含该月大豆进口数据的字典
        """
        try:
            # 构建日期字符串
            date_str = f"{year}-{month:02d}"
            
            # 获取所有大豆类型的进口数据
            all_soybean_data = {}
            
            for product_name, hs_code in self.soybean_hs_codes.items():
                data = self.get_soybean_import_data_by_hs_code(hs_code, date_str, date_str)
                if data:
                    all_soybean_data[product_name] = data
            
            # 计算总量
            total_import = 0
            for product_data in all_soybean_data.values():
                if product_data.get('import_data') and product_data['import_data'].get('quantity'):
                    total_import += product_data['import_data']['quantity']
            
            # 构建月度数据
            monthly_data = {
                'year': year,
                'month': month,
                'date_str': date_str,
                'total_import': total_import,
                'product_breakdown': all_soybean_data,
                'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return monthly_data
            
        except Exception as e:
            logger.error(f"获取 {year}年{month}月 大豆进口数据时出错: {str(e)}")
            return None
    
    def get_soybean_import_data_by_year(self, year: int) -> Optional[Dict]:
        """
        获取指定年份的大豆进口数据
        
        参数:
            year: 年份
            
        返回:
            包含该年大豆进口数据的字典
        """
        try:
            logger.info(f"正在获取 {year} 年大豆进口数据")
            
            # 获取该年所有月份的数据
            monthly_data = {}
            total_import = 0
            
            for month in range(1, 13):
                data = self.get_soybean_import_data_by_month(year, month)
                if data:
                    monthly_data[month] = data
                    total_import += data.get('total_import', 0)
                
                # 添加延迟避免请求过于频繁
                time.sleep(1)
            
            # 构建年度数据
            yearly_data = {
                'year': year,
                'total_import': total_import,
                'monthly_breakdown': monthly_data,
                'query_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return yearly_data
            
        except Exception as e:
            logger.error(f"获取 {year} 年大豆进口数据时出错: {str(e)}")
            return None
    
    def scrape_customs_announcement_data(self, start_date: str, end_date: str) -> Optional[List[Dict]]:
        """
        从海关总署公告中爬取大豆进口数据
        
        参数:
            start_date: 开始日期，格式YYYY-MM-DD
            end_date: 结束日期，格式YYYY-MM-DD
            
        返回:
            包含公告数据的列表
        """
        try:
            logger.info(f"正在爬取海关总署公告数据，时间范围: {start_date} 至 {end_date}")
            
            # 访问海关总署统计数据页面
            response = self.session.get(self.customs_urls['statistics'], timeout=30)
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
            
            # 过滤日期范围
            filtered_announcements = []
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            for announcement in announcements:
                if announcement['date']:
                    ann_dt = datetime.strptime(announcement['date'], '%Y-%m-%d')
                    if start_dt <= ann_dt <= end_dt:
                        # 获取公告详细内容
                        content = self._get_announcement_content(announcement['url'])
                        if content:
                            # 提取大豆进口数据
                            import_data = self._extract_soybean_data_from_content(content)
                            if import_data:
                                announcement['import_data'] = import_data
                                filtered_announcements.append(announcement)
            
            return filtered_announcements
            
        except Exception as e:
            logger.error(f"爬取海关总署公告数据时出错: {str(e)}")
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
            if 'monthly_breakdown' in data:  # 年度数据
                rows = []
                for month, month_data in data['monthly_breakdown'].items():
                    rows.append({
                        'year': data['year'],
                        'month': month,
                        'date': month_data['date_str'],
                        'total_import': month_data['total_import'],
                        'query_time': month_data['query_time']
                    })
                
                df = pd.DataFrame(rows)
            elif 'total_import' in data:  # 月度数据
                df = pd.DataFrame([{
                    'year': data['year'],
                    'month': data['month'],
                    'date': data['date_str'],
                    'total_import': data['total_import'],
                    'query_time': data['query_time']
                }])
            else:  # 其他类型数据
                df = pd.json_normalize(data)
            
            # 保存为CSV
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到 {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到CSV时出错: {str(e)}")
            return False
    
    def _get_product_name_by_hs_code(self, hs_code: str) -> str:
        """根据HS编码获取产品名称"""
        for name, code in self.soybean_hs_codes.items():
            if code == hs_code:
                return name
        return "未知产品"
    
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
                    date_str = '-'.join(parts[:3])
                return date_str
        
        return None
    
    def _get_announcement_content(self, url: str) -> Optional[str]:
        """获取公告内容"""
        try:
            response = self.session.get(url, timeout=30)
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


def main():
    """主函数"""
    # 创建数据获取器
    fetcher = ChinaCustomsDataFetcher()
    
    # 获取用户输入
    print("中国海关大豆进口数据获取工具")
    print("=" * 50)
    print("1. 按月份获取数据")
    print("2. 按年份获取数据")
    print("3. 爬取公告数据")
    
    choice = input("请选择获取方式 (1/2/3): ").strip()
    
    if choice == '1':
        # 按月份获取数据
        year = int(input("请输入年份 (例如: 2024): ").strip())
        month = int(input("请输入月份 (1-12): ").strip())
        
        data = fetcher.get_soybean_import_data_by_month(year, month)
        if data:
            filename = f"soybean_import_{year}_{month:02d}.csv"
            success = fetcher.save_data_to_csv(data, filename)
            if success:
                print(f"数据已成功保存到 {filename}")
                print(f"总进口量: {data['total_import']} 吨")
            else:
                print("数据保存失败")
        else:
            print("未能获取到数据")
    
    elif choice == '2':
        # 按年份获取数据
        year = int(input("请输入年份 (例如: 2024): ").strip())
        
        data = fetcher.get_soybean_import_data_by_year(year)
        if data:
            filename = f"soybean_import_{year}_annual.csv"
            success = fetcher.save_data_to_csv(data, filename)
            if success:
                print(f"数据已成功保存到 {filename}")
                print(f"年度总进口量: {data['total_import']} 吨")
            else:
                print("数据保存失败")
        else:
            print("未能获取到数据")
    
    elif choice == '3':
        # 爬取公告数据
        start_date = input("请输入开始日期 (YYYY-MM-DD): ").strip()
        end_date = input("请输入结束日期 (YYYY-MM-DD): ").strip()
        
        data = fetcher.scrape_customs_announcement_data(start_date, end_date)
        if data:
            filename = f"soybean_import_announcements_{start_date}_to_{end_date}.json"
            filepath = f"../data/raw/{filename}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"数据已成功保存到 {filename}")
            print(f"共获取到 {len(data)} 条公告数据")
        else:
            print("未能获取到数据")
    
    else:
        print("无效的选择")


if __name__ == "__main__":
    main()
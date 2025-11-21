import requests
import pandas as pd
from datetime import datetime
import time

# ================= 配置区域 =================
# 1. API 密钥 (你提供的 Primary Key)
API_KEY = 'c1606755a66549f0bd2ace08a14b29dd'

# 2. 基础 URL (UN Comtrade API v2)
BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

# 3. 参数设定
# HS Code 1201: 大豆
COMMODITY_CODE = '1201'
# 报告国: 中国 (156)
REPORTER_CODE = '156'
# 贸易伙伴: 阿根廷(32), 巴西(76), 美国(842)
PARTNER_CODES = '32,76,842'
# 贸易流向: M = Import (进口)
FLOW_CODE = 'M'

# 动态生成最近10年的年份字符串 (例如: 2014,2015...2023)
current_year = datetime.now().year
years = [str(y) for y in range(current_year - 11, current_year)] # 取过去10-11年，确保有数据
PERIOD_STR = ",".join(years)

def get_soybean_data():
    """
    获取大豆进口数据并返回 DataFrame
    """
    
    # 构建查询参数
    # 注意：API 对一次请求的数据量有限制，如果请求失败，可能需要按年份拆分请求
    params = {
        'reporterCode': REPORTER_CODE,
        'partnerCode': PARTNER_CODES,
        'cmdCode': COMMODITY_CODE,
        'flowCode': FLOW_CODE,
        'period': PERIOD_STR,
        'motCode': '0',          # 运输方式：所有
        'customsCode': 'C00',    # 关森制度：所有
        'includeDesc': 'true',   # 包含描述文本
    }

    headers = {
        'Ocp-Apim-Subscription-Key': API_KEY,
        'Accept': 'application/json'
    }

    print(f"正在请求 UN Comtrade API...")
    print(f"查询年份: {PERIOD_STR}")
    print(f"关注国家: 巴西, 美国, 阿根廷 -> 中国")
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        
        # 检查状态码
        if response.status_code != 200:
            print(f"Error: 请求失败，状态码 {response.status_code}")
            print(f"错误信息: {response.text}")
            return None

        data = response.json()
        
        # 提取数据列表
        if 'data' not in data or len(data['data']) == 0:
            print("未查询到数据，可能是年份太新官方尚未更新，或参数有误。")
            return None

        records = data['data']
        print(f"成功获取 {len(records)} 条记录。正在处理...")

        # 转换为 Pandas DataFrame 方便查看
        df = pd.DataFrame(records)

        # 筛选并重命名关键字段
        # period: 年份
        # partnerDesc: 伙伴国名称
        # netWgt: 净重 (公斤) -> 这是你最关心的“数量”
        # primaryValue: 贸易额 (美元)
        
        result_df = df[['period', 'partnerDesc', 'netWgt', 'primaryValue']].copy()
        
        # 数据清洗：将公斤转换为“万吨”方便阅读
        result_df['数量(万吨)'] = result_df['netWgt'] / 1000 / 10000
        result_df['金额(亿美元)'] = result_df['primaryValue'] / 100000000
        
        # 格式化显示
        pd.set_option('display.float_format', lambda x: '%.2f' % x)
        
        # 按年份和数量排序
        result_df = result_df.sort_values(by=['period', '数量(万吨)'], ascending=[True, False])
        
        return result_df

    except Exception as e:
        print(f"发生异常: {e}")
        return None

if __name__ == "__main__":
    df_result = get_soybean_data()
    
    if df_result is not None:
        print("\n====== 中国大豆进口数据 (来自 UN Comtrade) ======")
        print(df_result[['period', 'partnerDesc', '数量(万吨)', '金额(亿美元)']].to_string(index=False))
        
        print("\n提示：")
        print("1. netWgt 为原始公斤数，代码中已转换为万吨。")
        print("2. 数据源为中国向联合国申报的官方海关数据。")
    else:
        print("获取数据失败。")
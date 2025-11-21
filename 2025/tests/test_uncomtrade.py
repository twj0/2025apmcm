import requests
import pandas as pd
from datetime import datetime
import time

# ================= 配置区域 =================
# 1. API 密钥 (你提供的 Primary Key)
API_KEY = 'c1606755a66549f0bd2ace08a14b29dd'

# 2. 基础 URL (UN Comtrade API v2)
FREQ_CODE = 'M'  # monthly frequency
BASE_URL = f"https://comtradeapi.un.org/data/v1/get/C/{FREQ_CODE}/HS"

# 3. 参数设定
# HS Code 1201: 大豆
COMMODITY_CODE = '1201'
# 报告国: 中国 (156)
REPORTER_CODE = '156'
# 贸易伙伴: 阿根廷(32), 巴西(76), 美国(842)
PARTNER_CODES = '32,76,842'
# 贸易流向: M = Import (进口)
FLOW_CODE = 'M'

# 最近 20 年（含当前年），用于 monthly 频率查询
current_year = datetime.now().year
current_month = datetime.now().month
start_year = current_year - 19
BIRTH_MONTH_LIMIT = 10  # 2025 only first 10 months
YEARS_STR = ",".join(str(y) for y in range(start_year, current_year + 1))

# 测试模式开关 - 设为 True 只查询最近2年，便于调试
TEST_MODE = False

def get_soybean_data():
    """
    获取大豆进口数据并返回 DataFrame
    月度数据格式：对于 UN Comtrade API，月度数据的 period 参数需要指定具体月份，如 202301, 202302 等
    """
    
    # 基础查询参数（不含 period）
    base_params = {
        'reporterCode': REPORTER_CODE,
        'partnerCode': PARTNER_CODES,
        'cmdCode': COMMODITY_CODE,
        'flowCode': FLOW_CODE,
        'motCode': '0',          # 运输方式：所有
        'customsCode': 'C00',    # 关森制度：所有
        'includeDesc': 'true',   # 包含描述文本
    }

    headers = {
        'Ocp-Apim-Subscription-Key': API_KEY,
        'Accept': 'application/json'
    }

    # 测试模式下只查询最近2年
    freq_desc = "年度" if FREQ_CODE == 'A' else "月度"
    if TEST_MODE:
        test_start_year = current_year - 1
        print(f"【测试模式】正在请求 UN Comtrade API ({freq_desc})...")
        print(f"【测试模式】查询年份范围: {test_start_year} - {current_year}")
    else:
        test_start_year = start_year
        print(f"正在请求 UN Comtrade API ({freq_desc})...")
        print(f"查询年份范围: {start_year} - {current_year}")

    print(f"关注国家: 巴西, 美国, 阿根廷 -> 中国")
    
    all_records = []
    
    try:
        year_iter = range(test_start_year, current_year + 1)
        for year in year_iter:
            months = range(1, 11) if (year == current_year and current_month >= 10) else range(1, 13)
            if year == current_year and current_month < 10:
                months = range(1, min(10, current_month) + 1)

            period_list = [f"{year}{month:02d}" for month in months]

            # 如果测试模式，只保留最后 24 个月的periods以缩短测试时间
            if TEST_MODE:
                period_list = period_list[-12:]

            # chunk period_list into groups of max 12 months per request
            for i in range(0, len(period_list), 12):
                chunk = period_list[i:i+12]
                if not chunk:
                    continue
                period_chunk_str = ",".join(chunk)

                params = base_params.copy()
                params['period'] = period_chunk_str

                print(f"\n--- 查询 {year} ({chunk[0]}-{chunk[-1]}) ---")
                response = requests.get(BASE_URL, params=params, headers=headers, timeout=30)

                if response.status_code != 200:
                    print(f"Error: 请求失败，年份 {year}，状态码 {response.status_code}")
                    print(f"错误信息: {response.text}")
                    continue

                data = response.json()

                if 'data' not in data or len(data['data']) == 0:
                    print(f"年份 {year}、区间 {chunk[0]}-{chunk[-1]} 未返回数据。")
                    continue

                chunk_records = data['data']
                print(f"成功获取 {len(chunk_records)} 条记录（{chunk[0]}-{chunk[-1]}）。")
                all_records.extend(chunk_records)

                time.sleep(1)

        if not all_records:
            print("未查询到任何数据。")
            return None

        print(f"\n总计获取 {len(all_records)} 条记录。正在处理...")

        # 转换为 Pandas DataFrame 方便查看
        df = pd.DataFrame(all_records)

        # 筛选并重命名关键字段
        # period: 期别（在 monthly 频率下通常为 YYYYMM）
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
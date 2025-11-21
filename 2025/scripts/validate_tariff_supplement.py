import pandas as pd
import numpy as np

def validate_tariff_supplement():
    """验证关税数据补充结果"""
    
    # 读取补充后的数据
    df = pd.read_csv('data/processed/q1/q1_1_supplemented.csv')
    
    print('=== 关税数据补充验证 ===')
    print(f'总记录数: {len(df)}')
    print(f'时间范围: {df["period"].min()} - {df["period"].max()}')
    
    print('\n=== 关税税率分布 ===')
    rate_dist = df['tariff_rate'].value_counts().sort_index()
    print(rate_dist)
    
    print('\n=== 各国关税情况 ===')
    for country in ['USA', 'Brazil', 'Argentina']:
        country_data = df[df['partnerDesc'] == country]
        if len(country_data) > 0:
            print(f'\n{country}:')
            print(f'  记录数: {len(country_data)}')
            print(f'  税率范围: {country_data["tariff_rate"].min()} - {country_data["tariff_rate"].max()}')
            print(f'  税率分布:')
            rate_dist_country = country_data['tariff_rate'].value_counts().sort_index()
            for rate, count in rate_dist_country.items():
                print(f'    {rate*100:.0f}%: {count}条记录')
    
    # 检查关键时间节点的税率变化
    print('\n=== 关键时间节点验证 ===')
    test_periods = ['201806', '201807', '201901', '202001', '202408', '202504']
    for period in test_periods:
        if period in df['period'].values:
            period_data = df[df['period'] == period]
            usa_data = period_data[period_data['partnerDesc'] == 'USA']
            if len(usa_data) > 0:
                rate = usa_data['tariff_rate'].iloc[0]
                print(f'{period}: USA关税率 = {rate*100:.0f}%')
    
    # 检查原始数据中的税率变化
    print('\n=== 原始税率变化检查 ===')
    original_df = pd.read_csv('data/processed/q1/q1_1.csv')
    
    # 检查2024年8月后的数据
    recent_data = original_df[original_df['period'] >= 202408]
    if len(recent_data) > 0:
        print(f'2024年8月后原始数据记录数: {len(recent_data)}')
        print('原始税率分布:')
        recent_rates = recent_data['tariff_rate'].value_counts().sort_index()
        print(recent_rates)
        
        # 对比补充前后的数据
        supplemented_recent = df[df['period'] >= 202408]
        print('\n补充后2024年8月后税率分布:')
        supplemented_rates = supplemented_recent['tariff_rate'].value_counts().sort_index()
        print(supplemented_rates)

if __name__ == "__main__":
    validate_tariff_supplement()
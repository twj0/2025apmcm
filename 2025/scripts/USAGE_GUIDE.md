# 中国海关大豆进口数据获取工具使用指南

## 概述

本工具集用于从中国海关总署官方渠道获取大豆进口数据，包括完整的数据获取、处理和保存功能。工具集包含多个Python脚本，适用于不同的使用场景和需求。

## 文件说明

### 1. 主要脚本

- **simple_customs_soybean_fetcher.py** - 简化版数据获取器，适用于基本的数据获取需求
- **ssl_customs_soybean_fetcher.py** - SSL兼容版数据获取器，解决了SSL证书验证问题
- **china_customs_soybean_fetcher.py** - 完整版数据获取器，包含更多高级功能

### 2. 测试脚本

- **simple_test_customs_fetcher.py** - 简化版工具的测试脚本
- **test_customs_soybean_fetcher.py** - 完整版工具的测试脚本

### 3. 文档

- **README_customs_data_fetcher.md** - 技术文档和API说明
- **USAGE_GUIDE.md** - 本使用指南

## 环境要求

- Python 3.8+
- uv 包管理器 (推荐) 或 pip
- 网络连接

## 安装依赖

### 使用uv (推荐)

```bash
# 确保已安装uv
uv --version

# 安装项目依赖
uv pip install -r requirements.txt
```

### 使用pip

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基本使用

```python
from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher

# 创建获取器实例
fetcher = SimpleCustomsDataFetcher()

# 获取指定月份的数据
data = fetcher.get_monthly_import_data(2024, 1)

# 保存数据到CSV
if data:
    fetcher.save_data_to_csv(data, "soybean_import_2024_01.csv")
    print(f"获取到数据: {data['total_import']} 吨")
else:
    print("未能获取到数据")
```

### 2. 命令行使用

```bash
# 进入脚本目录
cd 2025/scripts

# 运行简化版工具
python simple_customs_soybean_fetcher.py

# 按照提示输入年份和月份
```

### 3. 使用SSL兼容版

```python
from ssl_customs_soybean_fetcher import SSLCustomsDataFetcher

# 创建SSL兼容的获取器实例
fetcher = SSLCustomsDataFetcher()

# 获取数据
data = fetcher.get_monthly_import_data(2024, 1)
```

## 数据获取方法

### 1. 数据源

工具会尝试从以下数据源获取数据：

1. **海关统计数据门户** (https://stats.customs.gov.cn/)
2. **海关总署公告** (https://www.customs.gov.cn/)
3. **其他公开数据源** (可扩展)

### 2. 查询参数

- **HS编码**: 大豆类目编码为 '1201'，具体子类包括：
  - 种用大豆: '1201100000'
  - 黄大豆: '1201901001'
  - 黑大豆: '1201901002'
  - 青大豆: '1201901003'
  - 其他大豆: '1201901090'

- **时间范围**: 可查询任意月份的数据，但通常数据会有1-2个月的延迟发布

- **数据类型**: 进口数量 (单位: 吨)

### 3. 数据格式

获取的数据包含以下字段：

```json
{
    "year": 2024,
    "month": 1,
    "date_str": "2024-01",
    "total_import": 1000000,
    "source": "stats_portal",
    "raw_data": {...},
    "query_time": "2024-01-20 12:00:00"
}
```

## 高级用法

### 1. 批量获取多个月份数据

```python
from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher
import pandas as pd

# 创建获取器实例
fetcher = SimpleCustomsDataFetcher()

# 定义要查询的月份范围
start_year, start_month = 2023, 1
end_year, end_month = 2023, 12

# 存储所有数据
all_data = []

# 逐月获取数据
for year in range(start_year, end_year + 1):
    for month in range(start_month, end_month + 1):
        data = fetcher.get_monthly_import_data(year, month)
        if data:
            all_data.append(data)
            print(f"成功获取 {year}年{month}月 数据")
        else:
            print(f"未能获取 {year}年{month}月 数据")

# 保存为单个CSV文件
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv("soybean_import_batch.csv", index=False, encoding='utf-8-sig')
    print(f"批量数据已保存到 soybean_import_batch.csv")
```

### 2. 自定义数据源

```python
from simple_customs_soybean_fetcher import SimpleCustomsDataFetcher

class CustomDataFetcher(SimpleCustomsDataFetcher):
    def _fetch_from_custom_source(self, year, month):
        # 实现自定义数据源获取逻辑
        # 返回格式化后的数据
        return self._format_monthly_data(year, month, {'quantity': 1000000}, 'custom_source')
    
    def get_monthly_import_data(self, year, month):
        # 先尝试父类方法
        data = super().get_monthly_import_data(year, month)
        
        # 如果父类方法失败，尝试自定义源
        if not data:
            data = self._fetch_from_custom_source(year, month)
        
        return data

# 使用自定义获取器
fetcher = CustomDataFetcher()
data = fetcher.get_monthly_import_data(2024, 1)
```

### 3. 数据可视化

```python
import matplotlib.pyplot as plt
import pandas as pd

# 读取之前保存的数据
df = pd.read_csv("soybean_import_batch.csv")

# 创建时间序列图
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['total_import'], marker='o')
plt.title('中国大豆月度进口量')
plt.xlabel('日期')
plt.ylabel('进口量 (吨)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('soybean_import_trend.png')
plt.show()
```

## 故障排除

### 1. 常见问题

**问题**: SSL证书验证失败
```bash
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

**解决方案**: 使用SSL兼容版工具
```python
from ssl_customs_soybean_fetcher import SSLCustomsDataFetcher
fetcher = SSLCustomsDataFetcher()
```

**问题**: 网络连接超时
```bash
TimeoutError: [WinError 10060] 连接超时
```

**解决方案**: 
1. 检查网络连接
2. 增加超时时间
3. 使用代理设置

**问题**: 未能获取到数据

**解决方案**:
1. 检查目标月份是否已发布数据
2. 尝试获取更早期的数据
3. 检查海关网站是否有结构变化

### 2. 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)

fetcher = SimpleCustomsDataFetcher()
data = fetcher.get_monthly_import_data(2024, 1)
```

## 数据来源与法律声明

### 1. 数据来源

本工具获取的数据来源于中国海关总署官方发布的公开数据：

- 海关统计数据门户: https://stats.customs.gov.cn/
- 海关总署官方网站: https://www.customs.gov.cn/

### 2. 使用限制

- 本工具仅用于学术研究目的
- 请遵守海关总署网站的使用条款
- 商业用途请直接联系海关总署获取授权

### 3. 免责声明

- 本工具仅作为数据获取的辅助工具
- 数据准确性以海关总署官方发布为准
- 开发者不对因使用本工具产生的任何后果负责

## 技术支持

如遇到问题，请：

1. 检查本指南的故障排除部分
2. 查看日志文件 `china_customs_soybeans_simple.log`
3. 运行测试脚本确认环境配置
4. 提交Issue到项目仓库

## 更新日志

### v1.0.0 (2025-01-20)
- 初始版本发布
- 实现基本数据获取功能
- 支持多种数据源
- 提供SSL兼容版本
- 完整的测试覆盖

---

*最后更新: 2025-01-20*
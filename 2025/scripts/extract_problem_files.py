import json

# 加载数据完整性报告
with open('data_integrity_report.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== 数据完整性检查摘要 ===")
print(f"总文件数: {data['summary']['total_files']}")
print(f"有问题文件数: {data['summary']['files_with_issues']}")
print(f"问题类型分布: {data['summary']['issues_by_type']}")
print("\n建议:")
for rec in data['summary']['recommendations']:
    print(f"- {rec}")

print("\n=== 问题文件列表 ===")
problem_files = []
for file, details in data['detailed_results'].items():
    if details['issues'] or details['warnings']:
        problem_files.append(file)
        print(f"\n文件: {file}")
        if details['issues']:
            print(f"  严重问题: {details['issues']}")
        if details['warnings']:
            print(f"  警告: {details['warnings']}")

print(f"\n=== 缺失数据文件 ===")
missing_data_files = []
for file, details in data['detailed_results'].items():
    if details['missing_data']:
        missing_data_files.append(file)
        print(f"\n文件: {file}")
        for field, info in details['missing_data'].items():
            print(f"  字段 '{field}': {info['count']} 个缺失值 ({info['percentage']:.2f}%)")

print(f"\n=== 统计信息 ===")
print(f"有问题文件总数: {len(problem_files)}")
print(f"有缺失数据文件总数: {len(missing_data_files)}")

# 按问题类型分类
print("\n=== 按问题类型分类 ===")
issue_types = {}
for file, details in data['detailed_results'].items():
    if details['issues']:
        for issue in details['issues']:
            if isinstance(issue, dict):
                issue_type = issue.get('type', 'unknown')
            else:
                issue_type = 'general'
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(file)

for issue_type, files in issue_types.items():
    print(f"\n{issue_type} ({len(files)} 个文件):")
    for file in files:
        print(f"  - {file}")

# 按警告类型分类
print("\n=== 按警告类型分类 ===")
warning_types = {}
for file, details in data['detailed_results'].items():
    if details['warnings']:
        for warning in details['warnings']:
            warning_type = warning.split(':')[0] if ':' in warning else 'general'
            if warning_type not in warning_types:
                warning_types[warning_type] = []
            warning_types[warning_type].append((file, warning))

for warning_type, items in warning_types.items():
    print(f"\n{warning_type} ({len(items)} 个文件):")
    for file, warning in items:
        print(f"  - {file}: {warning}")
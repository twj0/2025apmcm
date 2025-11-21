# Q5: Transformer宏观经济预测技术指南（续）

## 3.2 政策评估框架

```python
class PolicyEvaluationFramework:
    """政策评估框架"""
    
    def __init__(self, trained_model):
        self.model = trained_model
        self.model.eval()
        
    def evaluate_policy_scenarios(self):
        """评估不同政策场景"""
        scenarios = {
            'baseline': {
                'tariff': 0.025,
                'subsidy': 0,
                'retaliation': 0
            },
            'reciprocal_tariff': {
                'tariff': 0.20,
                'subsidy': 0,
                'retaliation': 0.15
            },
            'chips_act_only': {
                'tariff': 0.025,
                'subsidy': 50,  # billion USD
                'retaliation': 0
            },
            'combined_policy': {
                'tariff': 0.15,
                'subsidy': 30,
                'retaliation': 0.10
            }
        }
        
        results = {}
        
        for name, scenario in scenarios.items():
            # 运行预测
            forecast = self._run_scenario(scenario)
            
            # 分析结果
            analysis = self._analyze_forecast(forecast, scenario)
            
            results[name] = {
                'forecast': forecast,
                'analysis': analysis,
                'metrics': self._compute_metrics(forecast)
            }
        
        # 比较分析
        comparison = self._compare_scenarios(results)
        
        return {
            'scenarios': results,
            'comparison': comparison,
            'recommendations': self._generate_recommendations(comparison)
        }
    
    def _run_scenario(self, scenario):
        """运行单个场景"""
        # 准备输入数据
        historical = self._get_historical_data()
        policy = self._encode_scenario(scenario)
        
        with torch.no_grad():
            # 模型预测
            output = self.model(historical, policy_variables=policy)
            
            # 提取预测结果
            forecast = {
                'gdp': output['predictions'][:, :, 0],
                'inflation': output['predictions'][:, :, 1],
                'unemployment': output['predictions'][:, :, 2],
                'manufacturing': output['predictions'][:, :, 3],
                'trade_balance': output['predictions'][:, :, 4],
                'uncertainty': output['uncertainty']
            }
        
        return forecast
    
    def _analyze_forecast(self, forecast, scenario):
        """分析预测结果"""
        analysis = {}
        
        # GDP影响
        gdp_impact = forecast['gdp'].mean() - self.baseline_gdp
        analysis['gdp_impact'] = {
            'mean': gdp_impact,
            'cumulative': gdp_impact.sum(),
            'peak': forecast['gdp'].max(),
            'trough': forecast['gdp'].min()
        }
        
        # 通胀影响
        inflation_impact = forecast['inflation'].mean() - self.baseline_inflation
        analysis['inflation_impact'] = {
            'mean': inflation_impact,
            'peak': forecast['inflation'].max(),
            'duration_above_target': (forecast['inflation'] > 0.02).sum()
        }
        
        # 制造业回流
        manufacturing_change = forecast['manufacturing'][-1] - forecast['manufacturing'][0]
        analysis['reshoring'] = {
            'total_change': manufacturing_change,
            'rate': manufacturing_change / len(forecast['manufacturing']),
            'effectiveness': manufacturing_change / scenario.get('tariff', 1)
        }
        
        return analysis
```

## 4. 实施路线图

### 第一阶段（24小时）
1. 基础Transformer架构实现
2. 数据预处理pipeline
3. 简单训练循环

### 第二阶段（48小时）
1. 因果推断模块集成
2. 多尺度建模
3. 不确定性量化

### 第三阶段（72小时）
1. 政策评估框架
2. 制造业回流专项分析
3. 可视化和报告生成

## 5. 可视化与解释性

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

class TransformerVisualizer:
    """Transformer结果可视化"""
    
    def __init__(self):
        self.colors = sns.color_palette("husl", 7)
        
    def plot_forecast_with_uncertainty(self, forecast, variable_name):
        """绘制带不确定性的预测"""
        fig = go.Figure()
        
        # 均值预测
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast['mean']))),
            y=forecast['mean'],
            mode='lines',
            name='Mean Forecast',
            line=dict(color='blue', width=2)
        ))
        
        # 置信区间
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast['quantiles']['95%']))),
            y=forecast['quantiles']['95%'],
            mode='lines',
            name='95% CI Upper',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(forecast['quantiles']['5%']))),
            y=forecast['quantiles']['5%'],
            mode='lines',
            name='95% CI Lower',
            line=dict(width=0),
            fillcolor='rgba(0,100,200,0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'{variable_name} Forecast with Uncertainty',
            xaxis_title='Months Ahead',
            yaxis_title=variable_name,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_attention_heatmap(self, attention_weights):
        """绘制注意力热图"""
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(
            attention_weights,
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=['GDP', 'CPI', 'Unemployment', 'Industrial', 'Fed Rate', '10Y', 'SP500'],
            yticklabels=range(attention_weights.shape[0])
        )
        
        plt.title('Transformer Attention Patterns')
        plt.xlabel('Economic Variables')
        plt.ylabel('Time Steps')
        
        return plt.gcf()
    
    def plot_policy_comparison(self, scenarios_results):
        """比较不同政策场景"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = ['GDP Growth', 'Inflation', 'Unemployment', 
                  'Manufacturing Share', 'Trade Balance', 'Fiscal Revenue']
        
        for i, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            data = []
            labels = []
            
            for scenario_name, results in scenarios_results.items():
                data.append(results['metrics'][metric.lower().replace(' ', '_')])
                labels.append(scenario_name)
            
            ax.bar(labels, data, color=self.colors[:len(labels)])
            ax.set_title(metric)
            ax.set_ylabel('Impact (%)')
            ax.grid(True, alpha=0.3)
            
            # 添加基准线
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.suptitle('Policy Scenario Comparison', fontsize=16)
        plt.tight_layout()
        
        return fig
```

## 6. 预期成果

### 6.1 预测精度
- **短期预测（1-6个月）**: MAPE < 5%
- **中期预测（6-24个月）**: MAPE < 10%
- **方向准确率**: > 85%

### 6.2 政策洞察
- **关税影响量化**: GDP、通胀、就业的精确影响
- **制造业回流评估**: 回流速度和规模预测
- **最优政策组合**: 多目标优化下的政策建议

### 6.3 不确定性量化
- **预测区间**: 95%置信区间覆盖率 > 90%
- **风险评估**: 极端事件概率估计
- **稳健性分析**: 对模型假设的敏感性

## 7. 技术栈

```yaml
dependencies:
  - torch==2.0.1
  - transformers==4.31.0
  - pandas==2.0.3
  - numpy==1.24.3
  - matplotlib==3.7.2
  - seaborn==0.12.2
  - plotly==5.15.0
  - scipy==1.11.1
  - statsmodels==0.14.0
```

## 8. 模型部署与应用

```python
class MacroTransformerAPI:
    """模型API接口"""
    
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.preprocessor = DataPreprocessor()
        self.postprocessor = ResultPostprocessor()
        
    def predict(self, request):
        """API预测接口"""
        # 预处理输入
        processed = self.preprocessor.process(request['data'])
        
        # 模型推理
        with torch.no_grad():
            output = self.model(processed)
        
        # 后处理
        results = self.postprocessor.format(output)
        
        return {
            'status': 'success',
            'predictions': results,
            'metadata': {
                'model_version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def batch_predict(self, requests):
        """批量预测"""
        results = []
        
        for request in requests:
            result = self.predict(request)
            results.append(result)
        
        return results
```

## 9. 总结

Transformer架构为宏观经济预测提供了强大的工具：

1. **优势**:
   - 捕捉长程依赖关系
   - 多变量联合建模
   - 灵活的政策干预分析
   - 可解释的注意力机制

2. **创新点**:
   - 因果推断集成
   - 多尺度层次建模
   - 贝叶斯不确定性量化
   - 制造业回流专项分析

3. **应用价值**:
   - 为政策制定提供量化支持
   - 评估互惠关税的综合影响
   - 预测制造业回流前景
   - 识别经济风险和机遇

通过这套技术方案，可以为2025 APMCM Problem C提供高质量的宏观经济分析和政策评估工具。

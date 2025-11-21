# Q3: 图神经网络(GNN)半导体供应链分析技术指南

## 1. 问题背景与数据分析

### 1.1 数据现状
- **贸易数据**: 1368行贸易记录，覆盖228个贸易伙伴（2020-2025）
- **产出数据**: 高/中/低端芯片分段产出（2020-2024）
- **政策数据**: CHIPS法案补贴指数、对华出口管制强度
- **安全指标**: 自给率、对华依赖度、供应风险指数

### 1.2 供应链网络特征
- **节点类型**: 国家、企业、技术节点
- **边类型**: 贸易流、技术转移、投资关系
- **网络属性**: 高度不对称、存在关键节点、动态演化

## 2. GNN供应链建模框架

### 2.1 异构图构建

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, global_mean_pool
import networkx as nx
import pandas as pd
import numpy as np

class ChipSupplyChainGraph:
    """半导体供应链异构图构建器"""
    
    def __init__(self, trade_data, production_data, policy_data):
        self.trade_data = trade_data
        self.production_data = production_data
        self.policy_data = policy_data
        self.graph = HeteroData()
        
    def build_graph(self):
        """构建异构供应链图"""
        # 1. 创建节点
        self._create_nodes()
        
        # 2. 创建边
        self._create_edges()
        
        # 3. 添加节点特征
        self._add_node_features()
        
        # 4. 添加边特征
        self._add_edge_features()
        
        return self.graph
    
    def _create_nodes(self):
        """创建不同类型的节点"""
        # 国家节点
        countries = self.trade_data['partner_country'].unique()
        self.graph['country'].x = torch.zeros(len(countries), 64)  # 64维特征
        self.country_mapping = {c: i for i, c in enumerate(countries)}
        
        # 技术节点（高/中/低端）
        segments = ['high', 'mid', 'low']
        self.graph['segment'].x = torch.zeros(len(segments), 32)
        self.segment_mapping = {s: i for i, s in enumerate(segments)}
        
        # 企业节点（主要制造商）
        companies = self._identify_key_companies()
        self.graph['company'].x = torch.zeros(len(companies), 48)
        self.company_mapping = {c: i for i, c in enumerate(companies)}
        
    def _create_edges(self):
        """创建不同类型的边"""
        # 贸易关系边 (country -> country)
        trade_edges = self._extract_trade_edges()
        self.graph['country', 'trades_with', 'country'].edge_index = trade_edges
        
        # 生产关系边 (country -> segment)
        production_edges = self._extract_production_edges()
        self.graph['country', 'produces', 'segment'].edge_index = production_edges
        
        # 供应关系边 (company -> company)
        supply_edges = self._extract_supply_chain_edges()
        self.graph['company', 'supplies', 'company'].edge_index = supply_edges
        
        # 技术依赖边 (segment -> segment)
        tech_edges = self._extract_technology_edges()
        self.graph['segment', 'depends_on', 'segment'].edge_index = tech_edges
        
    def _add_node_features(self):
        """添加节点特征"""
        # 国家特征
        for country, idx in self.country_mapping.items():
            features = self._compute_country_features(country)
            self.graph['country'].x[idx] = torch.FloatTensor(features)
        
        # 技术段特征
        for segment, idx in self.segment_mapping.items():
            features = self._compute_segment_features(segment)
            self.graph['segment'].x[idx] = torch.FloatTensor(features)
        
        # 企业特征
        for company, idx in self.company_mapping.items():
            features = self._compute_company_features(company)
            self.graph['company'].x[idx] = torch.FloatTensor(features)
    
    def _compute_country_features(self, country):
        """计算国家节点特征"""
        country_data = self.trade_data[
            self.trade_data['partner_country'] == country
        ]
        
        features = []
        
        # 贸易特征
        features.append(country_data['chip_import_charges'].sum())  # 总进口额
        features.append(country_data['chip_import_charges'].mean())  # 平均进口额
        
        # 分段贸易
        for segment in ['high', 'mid', 'low']:
            segment_trade = country_data[
                country_data['segment'] == segment
            ]['chip_import_charges'].sum()
            features.append(segment_trade)
        
        # 政策特征
        if country == 'United States':
            features.extend([
                self.policy_data['subsidy_index'].mean(),
                self.policy_data['export_control_china'].mean()
            ])
        else:
            features.extend([0, 0])  # 其他国家暂无政策数据
        
        # 补齐到64维
        features.extend([0] * (64 - len(features)))
        
        return features[:64]
    
    def _identify_key_companies(self):
        """识别关键企业"""
        # 基于领域知识的关键企业列表
        return [
            'TSMC', 'Samsung', 'Intel', 'NVIDIA', 'AMD', 'Qualcomm',
            'Broadcom', 'ASML', 'Applied_Materials', 'SK_Hynix',
            'MediaTek', 'Infineon', 'STMicro', 'NXP', 'Renesas'
        ]
```

### 2.2 分层GNN架构

```python
class HierarchicalGNN(nn.Module):
    """分层图神经网络用于供应链分析"""
    
    def __init__(self, node_dims, hidden_dims=128, num_layers=3):
        super().__init__()
        
        # 节点嵌入层
        self.node_embeddings = nn.ModuleDict({
            'country': nn.Linear(node_dims['country'], hidden_dims),
            'segment': nn.Linear(node_dims['segment'], hidden_dims),
            'company': nn.Linear(node_dims['company'], hidden_dims)
        })
        
        # 异构图卷积层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('country', 'trades_with', 'country'): 
                    GATConv(hidden_dims, hidden_dims, heads=4, concat=False),
                ('country', 'produces', 'segment'): 
                    SAGEConv((hidden_dims, hidden_dims), hidden_dims),
                ('segment', 'depends_on', 'segment'): 
                    GATConv(hidden_dims, hidden_dims, heads=2, concat=False),
                ('company', 'supplies', 'company'): 
                    SAGEConv(hidden_dims, hidden_dims),
            }, aggr='mean')
            self.convs.append(conv)
        
        # 跨层注意力机制
        self.cross_attention = CrossLayerAttention(hidden_dims)
        
        # 任务特定头
        self.risk_predictor = RiskPredictionHead(hidden_dims)
        self.flow_predictor = FlowPredictionHead(hidden_dims)
        self.resilience_scorer = ResilienceScorer(hidden_dims)
        
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """前向传播"""
        # 1. 节点嵌入
        h_dict = {
            node_type: self.node_embeddings[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # 2. 图卷积
        layer_outputs = []
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: F.relu(h) for key, h in h_dict.items()}
            layer_outputs.append(h_dict)
        
        # 3. 跨层注意力融合
        h_dict = self.cross_attention(layer_outputs)
        
        # 4. 任务输出
        outputs = {
            'risk_scores': self.risk_predictor(h_dict),
            'flow_predictions': self.flow_predictor(h_dict, edge_index_dict),
            'resilience_scores': self.resilience_scorer(h_dict)
        }
        
        return outputs

class CrossLayerAttention(nn.Module):
    """跨层注意力机制"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
    def forward(self, layer_outputs):
        """融合多层输出"""
        # 堆叠各层输出
        stacked = {}
        for node_type in layer_outputs[0].keys():
            layers = torch.stack([
                layer[node_type] for layer in layer_outputs
            ], dim=0)  # [num_layers, num_nodes, hidden_dim]
            
            # 自注意力融合
            attended, _ = self.attention(layers, layers, layers)
            
            # 残差连接和层归一化
            output = layers[-1] + attended.mean(dim=0)
            stacked[node_type] = F.layer_norm(output, output.shape)
        
        return stacked
```

### 2.3 供应链风险评估模块

```python
class SupplyChainRiskAnalyzer:
    """供应链风险分析器"""
    
    def __init__(self, gnn_model):
        self.model = gnn_model
        self.risk_factors = {
            'concentration': ConcentrationRisk(),
            'geopolitical': GeopoliticalRisk(),
            'technology': TechnologyRisk(),
            'disruption': DisruptionRisk()
        }
        
    def comprehensive_risk_assessment(self, graph_data):
        """全面风险评估"""
        # 1. GNN推理
        with torch.no_grad():
            gnn_outputs = self.model(
                graph_data.x_dict,
                graph_data.edge_index_dict
            )
        
        # 2. 结构性风险指标
        structural_risks = self._compute_structural_risks(graph_data)
        
        # 3. 动态风险评分
        dynamic_risks = self._compute_dynamic_risks(gnn_outputs)
        
        # 4. 综合风险评分
        comprehensive_score = self._aggregate_risks(
            structural_risks, 
            dynamic_risks,
            gnn_outputs['risk_scores']
        )
        
        return {
            'overall_risk': comprehensive_score,
            'risk_breakdown': {
                'structural': structural_risks,
                'dynamic': dynamic_risks,
                'predicted': gnn_outputs['risk_scores']
            },
            'vulnerable_nodes': self._identify_vulnerable_nodes(graph_data),
            'critical_paths': self._find_critical_paths(graph_data),
            'resilience_metrics': gnn_outputs['resilience_scores']
        }
    
    def _compute_structural_risks(self, graph):
        """计算结构性风险指标"""
        G = self._to_networkx(graph)
        
        risks = {}
        
        # 中心性风险
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # 识别关键节点
        critical_nodes = [
            node for node, centrality in betweenness_centrality.items()
            if centrality > np.percentile(list(betweenness_centrality.values()), 90)
        ]
        
        risks['centralization'] = np.std(list(degree_centrality.values()))
        risks['critical_node_dependency'] = len(critical_nodes) / G.number_of_nodes()
        
        # 连通性风险
        if nx.is_connected(G.to_undirected()):
            risks['connectivity'] = 1.0 / nx.average_shortest_path_length(G.to_undirected())
        else:
            risks['connectivity'] = 0.0
        
        # 聚类风险
        clustering = nx.clustering(G.to_undirected())
        risks['clustering'] = np.mean(list(clustering.values()))
        
        return risks
    
    def _identify_vulnerable_nodes(self, graph):
        """识别脆弱节点"""
        vulnerabilities = {}
        
        # 使用PageRank识别重要节点
        G = self._to_networkx(graph)
        pagerank = nx.pagerank(G)
        
        # 计算节点鲁棒性
        for node in G.nodes():
            # 移除节点后的连通性损失
            G_temp = G.copy()
            G_temp.remove_node(node)
            
            if G.is_directed():
                before = nx.strongly_connected_components(G)
                after = nx.strongly_connected_components(G_temp)
            else:
                before = nx.connected_components(G)
                after = nx.connected_components(G_temp)
            
            # 计算影响
            impact = len(list(before)) - len(list(after))
            vulnerabilities[node] = {
                'importance': pagerank[node],
                'removal_impact': impact,
                'vulnerability_score': pagerank[node] * impact
            }
        
        # 排序返回最脆弱的节点
        sorted_vulnerable = sorted(
            vulnerabilities.items(),
            key=lambda x: x[1]['vulnerability_score'],
            reverse=True
        )
        
        return sorted_vulnerable[:10]  # 返回前10个最脆弱节点
```

### 2.4 贸易流预测与优化

```python
class TradeFlowOptimizer(nn.Module):
    """贸易流预测与优化模块"""
    
    def __init__(self, gnn_model, hidden_dim=128):
        super().__init__()
        self.gnn = gnn_model
        
        # 流量预测网络
        self.flow_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 优化目标网络
        self.objective_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 成本、风险、效率
        )
        
    def predict_trade_flows(self, graph_data, tariff_scenario):
        """预测给定关税情景下的贸易流"""
        # 添加关税信息到节点特征
        graph_data = self._add_tariff_features(graph_data, tariff_scenario)
        
        # GNN编码
        node_embeddings = self.gnn(
            graph_data.x_dict,
            graph_data.edge_index_dict
        )
        
        # 预测边的贸易流量
        edge_flows = {}
        for edge_type in graph_data.edge_index_dict:
            src_type, _, dst_type = edge_type
            edge_index = graph_data.edge_index_dict[edge_type]
            
            # 获取边的源和目标节点嵌入
            src_embeddings = node_embeddings[src_type][edge_index[0]]
            dst_embeddings = node_embeddings[dst_type][edge_index[1]]
            
            # 拼接嵌入并预测流量
            edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
            flows = self.flow_predictor(edge_features)
            
            edge_flows[edge_type] = flows
        
        return edge_flows
    
    def optimize_supply_chain(self, graph_data, constraints):
        """优化供应链配置"""
        # 定义优化问题
        optimizer = SupplyChainOptimizer(
            graph=graph_data,
            objective='minimize_risk',
            constraints=constraints
        )
        
        # 使用梯度下降优化
        optimal_config = optimizer.solve()
        
        return optimal_config

class SupplyChainOptimizer:
    """供应链优化求解器"""
    
    def __init__(self, graph, objective, constraints):
        self.graph = graph
        self.objective = objective
        self.constraints = constraints
        
    def solve(self):
        """求解优化问题"""
        # 初始化决策变量（贸易流量）
        flow_variables = self._initialize_variables()
        
        # 定义目标函数
        if self.objective == 'minimize_risk':
            objective_fn = self._risk_objective
        elif self.objective == 'minimize_cost':
            objective_fn = self._cost_objective
        else:
            objective_fn = self._multi_objective
        
        # 优化循环
        optimizer = torch.optim.Adam([flow_variables], lr=0.01)
        
        for iteration in range(1000):
            optimizer.zero_grad()
            
            # 计算目标值
            obj_value = objective_fn(flow_variables)
            
            # 添加约束惩罚
            penalty = self._compute_constraint_penalty(flow_variables)
            
            loss = obj_value + penalty
            loss.backward()
            optimizer.step()
            
            # 投影到可行域
            with torch.no_grad():
                flow_variables.data = self._project_to_feasible(flow_variables.data)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Objective = {obj_value.item():.4f}")
        
        return flow_variables.detach()
    
    def _risk_objective(self, flows):
        """风险最小化目标"""
        # 计算供应集中度风险
        concentration = self._compute_concentration(flows)
        
        # 计算断链风险
        disruption_risk = self._compute_disruption_risk(flows)
        
        return concentration + 0.5 * disruption_risk
    
    def _compute_concentration(self, flows):
        """计算供应集中度（HHI指数）"""
        # 归一化流量
        normalized_flows = F.softmax(flows, dim=0)
        
        # 计算HHI
        hhi = torch.sum(normalized_flows ** 2)
        
        return hhi
```

### 2.5 政策影响分析

```python
class PolicyImpactAnalyzer:
    """政策影响分析器"""
    
    def __init__(self, gnn_model, historical_data):
        self.model = gnn_model
        self.historical_data = historical_data
        self.counterfactual_generator = CounterfactualGenerator()
        
    def analyze_policy_impact(self, policy_scenario):
        """分析政策影响"""
        results = {}
        
        # 1. 基准情景
        baseline = self._run_baseline_scenario()
        results['baseline'] = baseline
        
        # 2. 政策情景
        policy_outcome = self._run_policy_scenario(policy_scenario)
        results['policy'] = policy_outcome
        
        # 3. 反事实分析
        counterfactuals = self.counterfactual_generator.generate(
            baseline, policy_scenario
        )
        results['counterfactuals'] = counterfactuals
        
        # 4. 差异分析
        impacts = self._compute_impacts(baseline, policy_outcome)
        results['impacts'] = impacts
        
        # 5. 敏感性分析
        sensitivity = self._sensitivity_analysis(policy_scenario)
        results['sensitivity'] = sensitivity
        
        return results
    
    def _run_policy_scenario(self, policy):
        """运行政策情景模拟"""
        # 构建政策调整后的图
        adjusted_graph = self._apply_policy_to_graph(
            self.historical_data, policy
        )
        
        # GNN预测
        with torch.no_grad():
            predictions = self.model(
                adjusted_graph.x_dict,
                adjusted_graph.edge_index_dict
            )
        
        # 提取关键指标
        metrics = {
            'self_sufficiency': self._compute_self_sufficiency(predictions),
            'supply_risk': predictions['risk_scores'].mean().item(),
            'trade_volume': self._compute_trade_volume(predictions),
            'technology_gap': self._compute_tech_gap(predictions),
            'economic_impact': self._compute_economic_impact(predictions)
        }
        
        return metrics
    
    def _sensitivity_analysis(self, base_policy):
        """敏感性分析"""
        sensitivity_results = {}
        
        # 定义参数范围
        param_ranges = {
            'tariff_rate': np.linspace(0, 0.5, 11),
            'subsidy_level': np.linspace(0, 50, 11),  # Billion USD
            'export_control': [0, 0.5, 1.0]  # 宽松、中等、严格
        }
        
        # 对每个参数进行扫描
        for param, values in param_ranges.items():
            param_sensitivity = []
            
            for value in values:
                # 修改政策参数
                test_policy = base_policy.copy()
                test_policy[param] = value
                
                # 运行模拟
                outcome = self._run_policy_scenario(test_policy)
                
                param_sensitivity.append({
                    'value': value,
                    'outcome': outcome
                })
            
            sensitivity_results[param] = param_sensitivity
        
        return sensitivity_results

class CounterfactualGenerator:
    """反事实情景生成器"""
    
    def generate(self, baseline, policy):
        """生成反事实情景"""
        counterfactuals = []
        
        # 情景1：如果没有CHIPS法案
        no_chips = self._remove_subsidies(policy)
        counterfactuals.append({
            'name': 'No CHIPS Act',
            'scenario': no_chips
        })
        
        # 情景2：如果中国不反制
        no_retaliation = self._remove_retaliation(policy)
        counterfactuals.append({
            'name': 'No Retaliation',
            'scenario': no_retaliation
        })
        
        # 情景3：技术突破情景
        tech_breakthrough = self._add_tech_breakthrough(policy)
        counterfactuals.append({
            'name': 'Tech Breakthrough',
            'scenario': tech_breakthrough
        })
        
        return counterfactuals
```

## 3. 训练与评估

### 3.1 训练流程

```python
class GNNTrainer:
    """GNN模型训练器"""
    
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5
        )
        
    def train(self, num_epochs=100):
        """训练主循环"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = self._train_epoch()
            
            # 验证阶段
            val_loss, val_metrics = self._validate()
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch)
            
            # 日志记录
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}")
            
            # 早停
            if self._should_early_stop(val_loss):
                print("Early stopping triggered")
                break
    
    def _train_epoch(self):
        """单个训练epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.data_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                batch.x_dict,
                batch.edge_index_dict
            )
            
            # 计算损失
            loss = self._compute_loss(outputs, batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.data_loader)
```

## 4. 实施路线图

### 第一阶段（24小时）
1. 供应链图数据构建
2. 基础GNN模型实现
3. 简单风险评分

### 第二阶段（48小时）
1. 异构图网络完整实现
2. 多任务学习框架
3. 政策影响分析工具

### 第三阶段（72小时）
1. 优化求解器集成
2. 反事实分析
3. 可视化仪表板

## 5. 预期成果

1. **网络洞察**: 识别供应链关键节点和脆弱环节
2. **风险量化**: 多维度供应链风险评分体系
3. **政策评估**: 关税/补贴/管制组合政策的效果预测
4. **优化方案**: 风险最小化的供应链配置建议

## 6. 技术栈

```yaml
dependencies:
  - torch==2.0.1
  - torch-geometric==2.3.1
  - networkx==3.1
  - pandas==2.0.3
  - numpy==1.24.3
  - scikit-learn==1.3.0
  - matplotlib==3.7.2
  - plotly==5.15.0
```

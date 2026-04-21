# MoFo-based Experiments

基于 [MoFo (Modulator + Forecast)](https://github.com/ChengqingYu/MoFo) 长期时序预测模型的改进实验，包含四个模型变体的对比研究。

## 模型概览

| 模型 | 周期偏置 (Periodic Bias) | 循环正则化 (Circulant Reg.) | 双分支分解 (Dual-Path Decomp.) |
|------|:---:|:---:|:---:|
| **MoFo** | ✅ | ❌ | ❌ |
| **MoFo_Circulant** | ❌ | ✅ | ❌ |
| **MoFo_CircBias** | ✅ | ✅ | ❌ |
| **MoFo_Circulant_DualPath** | ❌ | ✅ | ✅ |

### 各模型说明

- **MoFo**：原始基线模型，使用周期偏置 (Periodic Bias) 引入先验周期信息
- **MoFo_Circulant**：移除周期偏置，加入循环矩阵正则化 (Circulant Regularization)，使注意力矩阵趋向循环结构
- **MoFo_CircBias**：保留周期偏置的同时加入循环正则化，研究两者是否互补
- **MoFo_Circulant_DualPath**：在循环正则化基础上加入趋势-季节双分支分解，趋势用 MLP 处理，季节用 Attention 处理

## 文件结构

```
Mofo_based_exp/
├── README.md
├── models/                          # 模型核心代码
│   ├── MoFo.py                      # 原始 MoFo 模型
│   ├── MoFo_Circulant.py            # 循环正则化变体（无周期偏置）
│   ├── MoFo_CircBias.py             # 循环正则化 + 周期偏置变体
│   └── MoFo_Circulant_DualPath.py   # 循环正则化 + 双分支分解变体
├── adapters/                        # TFB 框架适配器
│   ├── adapters_for_MoFo.py                    # MoFo 适配器
│   ├── adapters_for_MoFo_Circulant.py          # MoFo_Circulant 适配器
│   ├── adapters_for_MoFo_CircBias.py           # MoFo_CircBias 适配器
│   └── adapters_for_MoFo_Circulant_DualPath.py # MoFo_Circulant_DualPath 适配器
├── scripts/                         # 实验脚本
│   └── run_compare.py               # 四模型对比实验脚本
├── baselines_init.py                # baselines/__init__.py 注册适配器
└── time_series_library_init.py      # time_series_library/__init__.py 注册模型
```

## 核心技术

### 1. 循环矩阵正则化 (Circulant Regularization)

**核心思想**：时间序列的注意力矩阵应具有近似循环结构（shift-invariance），即相同时间偏移的注意力权重应相似。

**实现要点**：
- 对注意力矩阵沿每条对角线求均值，构造循环近似矩阵
- 正则化损失：`L_reg = λ · MSE(Attn, CirculantApprox(Attn))`
- λ 通过 softplus 参数化，可学习
- Token Reweighting 机制：用可学习投影打破循环约束，处理非周期事件

### 2. 周期偏置 (Periodic Bias)

**核心思想**：在注意力分数上加入基于周期距离的偏置，使模型倾向于关注相同周期位置的 token。

**实现要点**：
- 偏置函数 `func()` 由 4 个低秩参数矩阵生成，输出为周期距离的平滑函数
- 周期距离 `diff` 为循环距离矩阵，不可学习
- 偏置以 `log(func())` 形式加到注意力分数上

### 3. 趋势-季节双分支分解 (Dual-Path Trend + Seasonal Decomposition)

**核心思想**：将时序信号分解为低频趋势和高频季节成分，分别用最适合的模块处理。

**实现要点**：
- **SeriesDecomp**：支持 FFT-STL（频域带通分离）和 MA（移动平均）两种模式
- **周期检测**：混合 FFT + ACF 两阶段——FFT 找能量峰值，ACF 验证自相关性
- **TrendProjector**：轻量级 MLP/Linear 处理趋势成分
- **Recomposer**：可学习 α 加权融合，`fused = α · trend + (1-α) · seasonal`
- 周期检测优先在原始时间序列上做（patch embedding 之前），频率分辨率更高

## 运行方式

将模型文件和适配器放入 [TFB (Time Series Forecasting Benchmark)](https://github.com/DecisionScienceLab/TFB) 框架的对应目录后运行：

```bash
python ./scripts/run_compare.py \
    --config-path "rolling_forecast_config.json" \
    --data-name-list "ETTh1.csv" \
    --gpus 0 \
    --num-workers 0 \
    --timeout 60000 \
    --save-path "ETTh1"
```

脚本会自动对四个模型分别以 seq_len = 96, 336, 512 运行实验，并在终端打印 MSE/MAE 对比表。

## 依赖

- Python >= 3.8
- PyTorch >= 1.12
- scikit-learn
- pandas
- numpy
- tqdm
- [TFB 框架](https://github.com/DecisionScienceLab/TFB)

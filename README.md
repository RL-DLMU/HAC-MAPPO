# 成品油强化学习调度框架

本项目使用强化学习解决成品油调度中的订单决策问题。以下内容介绍了项目的环境要求、配置方法以及使用方式。

---

## 项目环境与配置

### 环境要求
- **操作系统**：Windows/Linux/MacOS（已在Windows 11上测试）。
- **Python版本**：推荐使用Python 3.9及以上版本。
- **依赖库**：请参考`requirements.txt`安装所需的依赖库，主要依赖包括：
  - PyTorch
  - NumPy
  - Matplotlib
  - Math
  - Gym

### 配置方法
1. 确保已安装Python和相关依赖库。
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```
---

## 项目使用方式

### 训练模型
- 直接运行`main.py`即可：
  ```bash
  python HAC-MAPPO/envs/main.py
  ```
  训练过程中会记录指标到"HAC-MAPPO\envs\HAC-MAPPO.json"中，训练完成后模型保存在`HAC-MAPPO\trained_models`目录中。

## 数据集说明
本仓库提供了代码及运行环境。
对于真实场景中的数据配置，涉及实际加油站位置、库存及容量等保密数据以及版权问题，故不予公开。本仓库提供了合成数据的环境配置，其中油耗数据的生成方法已在论文实验部分的“油耗数据集描述”中进行了详细说明。


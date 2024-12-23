# CausalViz
CausalViz是一个强大的因果推断和可视化工具包，它能帮助研究者快速构建神经网络模型、分析变量间的因果关系，并提供直观的可视化结果。

## ✨ 核心特性
- 🔨 **自动化建模** - 基于神经网络的多变量拟合，自动保存拟合结果
  📊 **因果效应分析** - 一键计算和可视化任意变量的平均因果效应
  🎯 **调节效应分析** - 智能识别和量化变量间的调节作用
  📈 **统计检验** - 集成似然比检验，提供严谨的统计显著性分析
  🖼️ **可视化支持** - 自动生成清晰直观的分析图表

## 🔍 功能详解
### 1. 神经网络建模与拟合
基于pytorch机器学习框架，支持使用多个自变量X(x₁, x₂, x₃, ..., xₙ)对因变量y进行多次拟合建模。系统会自动保存所有拟合结果，为后续分析提供可靠的数据基础。
### 2. 平均效应分析
一键选择任意自变量xₖ，快速获取其对因变量y的平均效应拟合结果。同时集成似然比检验功能，可自动分析效应曲线各段的统计显著性，帮助研究者准确判断每一段变量间的因果关系强度。
### 3. 交互效应智能分析
计算其他自变量对目标变量xₖ的调节效应大小，并通过推荐最值得关注的分析方向，帮助研究者快速发现重要的变量交互作用。
### 4. 调节效应可视化
指定任意控制变量x_c，一键生成其对目标变量xₖ的调节效应分析图表。同时提供全局调节效应的似然比检验结果，直观展示控制变量的调节作用及其统计显著性。

## 🎨 效果展示
### 生成模拟数据
```python
n_samples = 5000  # 样本量
x1 = np.random.normal(0, 10, n_samples)      # 生成均值为0,标准差为10的正态分布
x2 = np.random.uniform(-10, 10, n_samples)   # 生成[-10,10]区间的均匀分布
x3 = np.random.normal(0, 10, n_samples)      # 生成均值为0,标准差为10的正态分布
x4 = np.random.uniform(-10, 10, n_samples)   # 生成[-10,10]区间的均匀分布
x5 = np.random.normal(-3, 1.5, n_samples)    # 生成均值为-3,标准差为1.5的正态分布
x8 = np.random.normal(0, 10, n_samples)      # 生成均值为0,标准差为10的正态分布
# 生成相关变量
x6 = x4 * 2                                  # x6与x4呈线性相关
x7 = x5 * 2 + np.random.normal(0, 0.3, n_samples)  # x7与x5呈线性相关,加入随机噪声
median_x4 = np.median(x4)                    # 计算x4的中位数
mask = x4 <= median_x4                       # 创建掩码,标记小于等于中位数的点
x4_new = np.random.uniform(-10, 0, int(n_samples/2))  # 生成新的均匀分布数据
x4[mask] = x4_new                           # 替换x4中较小的一半数据点
# 添加随机噪声
noise = np.random.normal(0, 5, n_samples)    # 生成随机噪声
# 生成因变量y,包含复杂的非线性关系和交互效应
y = (x3+10)/20*10*x1 + (10-x3)/20*x1**2 + (x2/2)**2 + (x4/2)**2 + 2*x5 + x6 + x7 + 2*x8 + noise
```
### 1. 平均因果效应示例图

![平均因果效应分析](https://github.com/user-attachments/assets/54b207b6-42a5-405f-9a4a-0e30f30ad007)

图1中元素含义：
- 散点坐标：每个点代表一个样本的[x1,y]值
- 蓝色热力图：表示拟合空间在y-x1平面的投影分布
- 红色曲线：展示x1对y的平均因果效应
            曲线粗细反映附近散点密度，散点越少曲线越细，表示该区域的拟合结果越不稳健

图2中元素含义：
- 红-蓝曲线：展示x1对y的平均因果效应
            曲线颜色从蓝到红表示 弯折指数 的大小，越红表示弯折越显著
            弯折指数 = 该点弯折的概率密度 × 该点弯折的幅度，用于量化每个点处弯折的显著程度
            弯折指数较大的点可能代表重要的门限值，是变量关系发生显著变化的位置
- 黄色曲线：展示神经网络在各点处出现弯折的概率密度

图3中元素含义：
- 散点坐标：每个点代表一个样本的[x1,y_pred+residual]值
- 红色曲线：展示x1对y的平均因果效应
            开启计算LRT统计量后可以计算每一段的斜率及其显著性
- 灰色曲线：表示残差随x1变化的走势（局部残差/总体残差）
下图为开启计算LRT统计量后的效果：
![平均因果效应分析：开启LRT统计量计算](https://github.com/user-attachments/assets/b696642d-0441-46cb-8680-55e3a16941ef)

### 2. 调节效应（交互效应）指数分析

![调节效应（交互效应）指数分析](https://github.com/user-attachments/assets/5729d300-1776-4ea6-919b-fff3a007557e)

该指数已归一化处理，取值范围为[0,1]，取值越大表示调节效应越显著
可以看到x3对x1的调节效应最大，符合我们数据的生成过程

### 3. 调节效应（交互效应）可视化

![调节效应（交互效应）可视化](https://github.com/user-attachments/assets/2bcdd537-32e3-4e9d-929c-6a3cd88c623c)

图中元素含义：
- 该图固定画出五条曲线，表示x3在不同取值的情况下，x1对y的平均因果效应
同时该图支持计算LRT统计量，提供x3对x1的调节效应的全局检验，下图为开启LRT统计量计算后的效果：

![调节效应（交互效应）可视化：开启LRT统计量](https://github.com/user-attachments/assets/abe93ea0-6591-4e82-9b18-e54fcdd53097)

对比：x4对x1的调节效应显著性检验

![x4对x1的调节效应显著性检验](https://github.com/user-attachments/assets/ce737f07-a273-4a54-b1f8-4eafa3a306dc)


根据数据的生成过程可知，x4对x1的调节效应不显著，符合我们的预期

## 🚀 快速开始

### 环境要求
- Python >= 3.7
 PyTorch >= 1.8.0
 NumPy >= 1.19.2
 Pandas >= 1.2.3
 Matplotlib >= 3.3.4
### 安装方法
pip install CausalViz
### 使用方法
```python
visualizer = CausalVisualizer()

visualizer.load_data(df,y_col='y')

visualizer.NN_singlefit(n=30,
    epochs=5000,
    learning_rate=0.1,
    min_lr=0.01,
    switch_threshold=0.01,
    patience=1000,
    min_delta=1e-7,
    decay_rate=[0.8, 50],
    penalty_strength=1)

multi_fit_result = visualizer.NN_multifit(25,
    trials=10,
    epochs=2000,
    learning_rate=0.1,
    min_lr=0.01,
    switch_threshold=0.01,
    patience=1000,
    min_delta=1e-7,
    decay_rate=[0.8, 50],
    penalty_strength=1,
    pop_ratio=0.5)

visualizer.plot_average_effects(multi_fit_result,
    x_name='x1',
    k=5000,
    caculate_significance=True,
    train_control_model_setting=[5, 1000, 0.1, [0.9, 50], 0.5],
    bboxtoanchor2=(0.5, 0.9),
    bboxtoanchor3=(0.5, 0.6))

visualizer.analyze_moderation_effect(multi_fit_result,
    x_name='x1',
    )
    
visualizer.plot_moderation_effect(multi_fit_result,
    main_x_name='x1',
    control_x_name='x4',
    caculate_moderation_LRT=True,
    train_control_model_setting=[5, 2000, 0.1, 0.01, 0.01, 1000, 1e-7, [0.8, 50], 0.5]
)
```






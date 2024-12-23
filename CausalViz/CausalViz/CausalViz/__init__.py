"""
CausalViz - 低维度因果分析可视化工具

一个用于进行低维度因果分析和可视化的Python库。
"""

__version__ = "0.1.0"
__author__ = "xiezhenyuan-dashuaibi"
__author_email__ = "546091915@qq.com"
__license__ = "MIT"
__description__ = "低维度因果分析可视化工具"
__url__ = "https://github.com/xiezhenyuan-dashuaibi/CausalViz"


import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from seaborn import kdeplot
import torch
import torch.nn as nn
import matplotlib.pyplot as plt




class CausalVisualizer:
    def __init__(self):
        """初始化因果推断可视化器"""
        self.model = None
        self.fig = None
        self.axes = None
        self.X = None
        self.y = None
        self.y_col = None  # 保存y的列名
        self.X_cols = None  # 保存X的列名
        self.X_tensor = None  # 保存转换后的X张量
        self.y_tensor = None  # 保存转换后的y张量
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.loss_history = []
        self.n_thresholds_dict = {}
        self.max_threshold_stability_list = []
        

        
    def load_data(self, df, y_col):
        """
        加载数据
        
        参数:
            df: pandas DataFrame, 包含特征和目标变量的数据框
            y_col: str, 目标变量的列名
        """
        # 检查是否存在缺失值
        if df.isnull().any().any():
            raise ValueError("数据中存在缺失值,请先去除包含缺失值的行后再进行分析。")


        # 保存列名
        self.y_col = y_col
        self.X_cols = df.drop(columns=[y_col]).columns.tolist()
        
        # 分离特征和目标变量
        self.y = df[y_col].values.reshape(-1, 1)
        self.X = df.drop(columns=[y_col]).values

        print("原数据展示:")
        print(self.X_cols)
        print(self.X)
        print(self.y_col)
        print(self.y)
        
        # 保存每个变量的范围（标准化前）
        self.X_ranges = {
            col: (df[col].min(), df[col].max()) 
            for col in self.X_cols
        }

        # 保存标准化参数
        self.X_mean = self.X.mean(axis=0)
        self.X_std = self.X.std(axis=0)
        self.y_mean = self.y.mean()
        self.y_std = self.y.std()
        
        # 标准化数据
        Xs = (self.X - self.X_mean) / self.X_std
        ys = (self.y - self.y_mean) / self.y_std

        self.X_ranges_standardized = {
            col: ((min_val - mean) / std, (max_val - mean) / std)
            for (col, (min_val, max_val)), mean, std 
            in zip(self.X_ranges.items(), self.X_mean, self.X_std)
        }
        
        self.X_tensor = torch.FloatTensor(Xs)
        self.y_tensor = torch.FloatTensor(ys)
        

        print("标准化后数据展示:")
        print(self.X_tensor)
        print(self.y_tensor)

        print("\n原始数据范围:")
        for col, (min_val, max_val) in self.X_ranges.items():
            print(f"{col}: [{min_val:.4f}, {max_val:.4f}]")
        
        print("\n标准化后的范围:")
        for col, (min_val, max_val) in self.X_ranges_standardized.items():
            print(f"{col}: [{min_val:.4f}, {max_val:.4f}]")
        
    def NN_mse_stability_analysis(self, min_neurons=1, trials=40, epochs=5000, learning_rate=0.1, batch_ratio=0.5,min_lr=0.01,switch_threshold=0.01,patience=1000,min_delta=1e-7,decay_rate=[0.8,50],penalty_strength=1,pop_ratio=0.75):
        """
        分析不同神经元数量对模型性能的影响
        1. 数据量较大时，可以先用较小的比例快速测试
        2. 使用较小的数据比例时，建议适当增加trials数量
        3. 最终模型训练时可以使用全部数据以获得最佳效果
        
        参数:
            基础参数：
            start_neurons: int, 起始神经元数量
            min_neurons: int, 最小神经元数量
            trials: int, 每个神经元数量的训练次数，至少为2
            epochs: int, 每次训练的轮次
            learning_rate: float, 学习率
            decay_rate: list, 第一阶段的学习率衰减参数，[factor, patience]，factor为衰减一次学习率乘以的因子，patience为无改善多少轮后衰减一次
            batch_ratio: float, 选择使用样本量的比例进行训练，当batch_ratio=1时，表示使用全部样本量进行训练；若样本量较大，则可以适当调小batch_ratio，建议的最小batch_ratio为（30*变量数*神经元数量/样本量）
            min_lr: float, 最小学习率
            switch_threshold: float, 切换阈值
            patience: int, 早停耐心值，连续patience轮都没有超过最佳损失，则早停
            min_delta: float, 最小改善阈值，用于判断是否出现显著改善，若改善超过该值则记录下来。该值较大程度决定了早停的时机，可设置为0.001左右
            pop_ratio: float, 每次删除的训练最差的trial比例，至少删1条,当训练过程loss不稳定的时候可以提高，建议trials增多（10次以上），pop_ratio设为0.5以上
            stage1为reduceLROnPlateau阶段，stage2为余弦退火阶段

        """
        import copy
        if not 0 < batch_ratio <= 1:
            raise ValueError("batch_ratio必须在0到1之间")
        # 检查trials参数是否至少为2
        if trials < 2:
            raise ValueError("trials参数必须大于等于2,因为需要至少2次训练才能计算折点稳定性")
        print(f"建议的最大神经元数量不超过{len(self.X_cols)*3}")
        
        start_neurons = int(input("请输入起始神经元数量(建议不超过上述数值):"))
        n = start_neurons
        mse_list = []  # 用于存储每次训练的MSE
        threshold_stability_list = [] 
        self.thresholds_dict = {}
        self.n_thresholds_dict = {}
        self.max_threshold_stability_list = []

        # 设置批量大小和批次数
        batch_size = min(512, len(self.X))
        total_batches = len(self.X) // batch_size + (1 if len(self.X) % batch_size != 0 else 0)
        nbatches = max(1, int(total_batches * batch_ratio))  # 确保至少有1个批次
        
        print(f"数据总量: {len(self.X)}")
        print(f"批次大小: {batch_size}")
        print(f"总批次数: {total_batches}")
        print(f"使用数据比例: {batch_ratio*100}%")
        print(f"实际使用批次数: {nbatches}")
        
        while n >= min_neurons:
            print(f"\n使用 {n} 个神经元训练模型")
            trial_mses = []
            
            trial = 0
            while trial < trials:
                print(f"第 {trial+1} 次训练")
                
                # 创建模型
                input_size = len(self.X_cols)
                self.create_network(input_size=input_size, hidden_size=n, output_size=1)
                # 随机打乱数据
                perm = torch.randperm(len(self.X_tensor))
                batch_indices = [(i*batch_size, min((i+1)*batch_size, len(self.X))) for i in range(nbatches)]
                X_batches = [self.X_tensor[perm[start:end]] for start, end in batch_indices]
                y_batches = [self.y_tensor[perm[start:end]] for start, end in batch_indices]
                # 定义损失函数和优化器
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=learning_rate,
                    weight_decay=0
                )
                # 第一阶段：使用ReduceLROnPlateau
                scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=decay_rate[0],
                    patience=decay_rate[1],
                    verbose=False,
                    min_lr=min_lr
                )
                
                # 记录损失历史
                recent_losses = []
                switch_epoch = None
                stage = 1  # 标记当前阶段
                best_model = None
                best_loss = float('inf')
                epochs_since_best = 0  # 初始化无改善轮数计数
                need_retrain = False # 标记是否需要重新训练
                
                #训练模型
                for epoch in range(epochs):
                    epoch_loss = 0.0
                    
                    for b in range(total_batches):
                        optimizer.zero_grad()
                        outputs, penalty = self.model(X_batches[b])  
                        loss = criterion(outputs, y_batches[b]) + penalty_strength * penalty  # 添加惩罚项
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    epoch_loss /= nbatches
                    recent_losses.append(epoch_loss)

                    # 在第100个epoch检查loss
                    if epoch == 99 and epoch_loss > 10:
                        need_retrain = True
                        print(f"第100轮loss为{epoch_loss:.4f}，大于10，需要重新训练")
                        break

                    # 更新最佳模型，添加最小改善阈值判断
                    if epoch_loss < (best_loss - min_delta):
                        best_loss = epoch_loss
                        best_model = copy.deepcopy(self.model)
                        epochs_since_best = 0  # 重置计数器
                    else:
                        epochs_since_best += 1
                    
                    # 第一阶段：检查是否需要切换到余弦退火
                    if stage == 1:
                        scheduler1.step(epoch_loss)
                        
                        # 检查最近50个epoch的变化
                        if len(recent_losses) >= 107:
                            recent_change = abs(np.mean(recent_losses[-6:-1]) - np.mean(recent_losses[-106:-101])) / np.mean(recent_losses[-106:-101])
                            if recent_change < switch_threshold:  
                                switch_epoch = epoch
                                stage = 2
                                print(f"\n在第 {epoch+1} 轮切换到余弦退火学习率调度器")
                                # 直接调整优化器的学习率
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = learning_rate
                                
                                # 创建余弦退火调度器，从新的最大学习率开始
                                scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                                    optimizer,
                                    T_max=switch_epoch,  # 用找到的epoch数作为周期
                                    eta_min=min_lr  
                                )
                    
                    # 第二阶段：使用余弦退火
                    else:
                        scheduler2.step()

                    # 打印训练信息
                    if (epoch + 1) % 100 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, '
                              f'LR: {current_lr:.6f}, Stage: {stage}, '
                              f'Epochs since best: {epochs_since_best}')
                        
                    # 如果连续patience轮都没有超过最佳损失，则早停
                    if epochs_since_best >= patience:
                        print(f'连续{epochs_since_best}轮未出现显著改善(改善阈值:{min_delta})，在第{epoch+1}轮提前停止训练')
                        break

                # 如果需要重新训练，跳过后续步骤，重新开始这次trial
                if need_retrain:
                    continue

                self.loss_history = []
                # 使用最佳模型评估
                self.model = best_model
                with torch.no_grad():
                    y_pred,_ = self.model(self.X_tensor) 
                    mse = criterion(y_pred, self.y_tensor).item()
                    trial_mses.append(mse)
                    print(f"第 {trial+1} 次训练的MSE: {mse:.4f}")
                    
                    # 将模型参数存储为矩阵
                    params_dict = {}
                    for name, param in self.model.named_parameters():
                        params_dict[name] = param.data.numpy()
                    
                    print("\n模型参数:")
                    for name, param_matrix in params_dict.items():
                        print(f"{name}:")
                        print(param_matrix)
                    
                    # 获取参数
                    hidden_weight = params_dict['hidden.weight'].copy()
                    hidden_bias = params_dict['hidden.bias']

                    # 为每个输入变量计算折点
                    # 创建当前trial的字典
                    trial_thresholds = {}
                    for i in range(len(self.X_cols)):
                        col_weights = hidden_weight[:, i]  # 获取第i个输入变量的权重
                        thresholds = -hidden_bias / col_weights  # 计算折点
                        sorted_thresholds = np.sort(thresholds)  # 对折点排序
                        # 计算中值
                        x_min, x_max = self.X_ranges_standardized[self.X_cols[i]]
                        mid_point = (x_min + x_max) / 2
                        # 计算每个折点与中值的绝对差
                        sorted_thresholds = np.abs(sorted_thresholds - mid_point)
                        trial_thresholds[self.X_cols[i]] = sorted_thresholds  # 保存到字典中
                    
                    # 将当前trial的结果存入全局字典
                    self.thresholds_dict[trial] = trial_thresholds

                # 只有当这次trial成功完成时，才增加trial计数
                trial += 1

            self.n_thresholds_dict[n] = self.thresholds_dict.copy()
            self.thresholds_dict = {}
                
            # 计算要删除的数量(至少删1条)
            num_to_remove = max(1, int(len(trial_mses) * pop_ratio))
            
            # 找出最大的几个MSE对应的trial索引
            sorted_indices = sorted(range(len(trial_mses)), key=lambda k: trial_mses[k], reverse=True)
            indices_to_remove = sorted_indices[:num_to_remove]
            
            # 从大到小依次删除
            for idx in sorted(indices_to_remove, reverse=True):
                # 删除MSE
                trial_mses.pop(idx)
                
                # 删除对应的thresholds_dict条目
                if idx in self.n_thresholds_dict[n]:
                    del self.n_thresholds_dict[n][idx]

            remaining_trials = len(trial_mses)  # 获取剩余的trial数量
            avg_mse = sum(trial_mses) / remaining_trials
            min_mse = min(trial_mses)  
            harmonic_mean_mse = 1/sum([1/x for x in trial_mses])  # 使用调和平均值，给小的MSE更大的权重
            mse_list.append((n, avg_mse, min_mse, harmonic_mean_mse))
            print(f"\n使用 {n} 个神经元的平均MSE: {avg_mse:.4f}")

            # 为每个输入变量计算随机抽样的折点差异
            for i, col in enumerate(self.X_cols):
                # 获取该变量的标准化范围
                x_min, x_max = self.X_ranges_standardized[col]
                range_width = x_max - x_min
                
                # 定义软边界(范围的10%缓冲)
                soft_min = x_min + 0.1 * range_width  
                soft_max = x_max - 0.1 * range_width
                

                random_thresholds = []
                mid_point = (soft_min + soft_max) / 2  # 计算中值
                
                # 生成100个向量
                for _ in range(100):
                    # 在软边界内均匀抽取n个点并排序
                    points = np.sort(np.random.uniform(soft_min, soft_max, n))
                    # 计算每个点与中值的绝对差
                    abs_diffs = np.abs(points - mid_point)
                    random_thresholds.append(abs_diffs)
                
                # 将random_thresholds转换为numpy数组以便计算
                random_thresholds = np.array(random_thresholds)
                
                # 计算每个位置的标准差
                stds = np.std(random_thresholds, axis=0)
                
                # 对每个向量的每个元素进行标准化
                random_thresholds = random_thresholds / stds
                
                # 对n_threshold_dict中的trial_threshold进行标准化
                for trial_idx in self.n_thresholds_dict[n].keys():
                    self.n_thresholds_dict[n][trial_idx][col] = self.n_thresholds_dict[n][trial_idx][col] / stds
                
                # 计算向量两两相减的模的均值
                total_diff = 0
                count = 0
                for j in range(100):
                    for k in range(j+1, 100):
                        # 计算两个向量的差的模
                        diff = np.linalg.norm(random_thresholds[j] - random_thresholds[k])
                        total_diff += diff
                        count += 1
                
                if count > 0:
                    avg_diff_random = (total_diff / count)/np.sqrt(n)#完全随机折点下的平均稳定性
                    self.max_threshold_stability_list.append((n, f"{col}_random", avg_diff_random))

            # 为每个输入变量计算折点差异
            for col in self.X_cols:
                # 提取当前变量的所有折点
                col_thresholds = []
                for trial_idx in self.n_thresholds_dict[n].keys():  # 只使用未被删除的trial
                    col_thresholds.append(self.n_thresholds_dict[n][trial_idx][col])
                
                # 计算向量两两相减的模的加权均值
                total_weighted_diff = 0
                total_weight = 0
                for i in range(len(col_thresholds)):
                    for j in range(i+1, len(col_thresholds)):
                        # 获取两个trial的mse作为权重
                        weight = 1/np.sqrt(trial_mses[i] * trial_mses[j])
                        # 计算两个向量的差的模并加权
                        diff = np.linalg.norm(col_thresholds[i] - col_thresholds[j])
                        total_weighted_diff += weight * diff
                        total_weight += weight

                if total_weight > 0:
                    avg_diff = (total_weighted_diff / total_weight)/np.sqrt(n)
                    # 获取当前变量的随机稳定性
                    random_stability = next(s[2] for s in self.max_threshold_stability_list if s[0] == n and s[1] == f"{col}_random")
                    # 计算相对稳定性
                    relative_stability = avg_diff / random_stability
                    threshold_stability_list.append((n, col, relative_stability))  # 将每个变量的稳定性除以其对应的随机稳定性,得到相对稳定性
            
            
            # 计算当前n下每个变量threshold_stability的root mean square和mean
            current_n_stabilities = [item for item in threshold_stability_list if item[0] == n]
            if current_n_stabilities:
                stability_squares = [stab[2]**2 for stab in current_n_stabilities]
                root_mean_square_stability = np.sqrt(sum(stability_squares) / len(stability_squares))
                mean_stability = sum(stab[2] for stab in current_n_stabilities) / len(current_n_stabilities)
                threshold_stability_list.append((n, "root mean square", root_mean_square_stability))
                threshold_stability_list.append((n, "mean", mean_stability))
            #root_mean_square给大的变量的折点稳定性更大的权重，这个值减小才说明整体的稳定性更好
            
            
            
            n -= 1
            
        return [mse_list,threshold_stability_list]
    def NN_singlefit(self, n, epochs=5000, learning_rate=0.1, min_lr=0.01, switch_threshold=0.01, patience=1000, min_delta=1e-7,decay_rate=[0.8,50],penalty_strength=1):
        """
        该函数用于预训练神经网络，探索调参方式，调好参数后，再使用NN_multifit/NN_mse_stability_analysis函数进行训练。
        训练神经网络，并返回训练过程。
        stage1为reduceLROnPlateau阶段，stage2为余弦退火阶段
        参数:
            基础参数：
            epochs: int, 每次训练的轮次
            learning_rate: float, 学习率
            min_lr: float, 最小学习率
            switch_threshold: float, 切换阈值
            patience: int, 早停耐心值，连续patience轮都没有超过最佳损失，则早停
            min_delta: float, 最小改善阈值，用于判断是否出现显著改善，若改善超过该值则记录下来。该值较大程度决定了早停的时机，可设置为0.001左右
            decay_rate: list, 第一阶段的学习率衰减参数，[factor, patience]，factor为衰减一次学习率乘以的因子，patience为无改善多少轮后衰减一次
            penalty_strength: float, 惩罚项强度，用于控制折点超出软边界的惩罚，默认1
        """
        import copy

        # 设置批量大小和批次数
        batch_size = min(512, len(self.X))
        total_batches = len(self.X) // batch_size + (1 if len(self.X) % batch_size != 0 else 0)

        # 准备数据批次
        indices = torch.randperm(len(self.X))
        X_batches = torch.split(self.X_tensor[indices], batch_size)
        y_batches = torch.split(self.y_tensor[indices], batch_size)

        # 创建模型
        input_size = len(self.X_cols)
        self.create_network(input_size=input_size, hidden_size=n, output_size=1)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0
        )
                
        # 第一阶段：使用ReduceLROnPlateau
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=decay_rate[0],
            patience=decay_rate[1],
            verbose=False,
            min_lr=min_lr
        )
                
        # 记录损失历史
        recent_losses = []
        switch_epoch = None
        stage = 1  # 标记当前阶段
        best_model = None
        best_loss = float('inf')
        epochs_since_best = 0  # 初始化无改善轮数计数

        # 训练模型
        for epoch in range(epochs):
            epoch_loss = 0.0
                    
            for b in range(total_batches):
                optimizer.zero_grad()
                outputs, penalty = self.model(X_batches[b])  
                loss = criterion(outputs, y_batches[b]) + penalty_strength * penalty  # 添加惩罚项
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                optimizer.step()
                        
                epoch_loss += loss.item()
                    
            epoch_loss /= total_batches
            recent_losses.append(epoch_loss)

            # 更新最佳模型，添加最小改善阈值判断
            if epoch_loss < (best_loss - min_delta):
                best_loss = epoch_loss
                best_model = copy.deepcopy(self.model)
                epochs_since_best = 0  # 重置计数器
            else:
                epochs_since_best += 1
                    
            # 第一阶段：检查是否需要切换到余弦退火
            if stage == 1:
                scheduler1.step(epoch_loss)
                        
                # 检查最近50个epoch的变化
                if len(recent_losses) >= 107:
                    recent_change = abs(np.mean(recent_losses[-6:-1]) - np.mean(recent_losses[-106:-101])) / np.mean(recent_losses[-106:-101])
                    if recent_change < switch_threshold:  
                        switch_epoch = epoch
                        stage = 2
                        print(f"\n在第 {epoch+1} 轮切换到余弦退火学习率调度器")
                        # 直接调整优化器的学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate

                        # 创建余弦退火调度器，从新的最大学习率开始
                        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=switch_epoch,  # 用找到的epoch数作为周期
                            eta_min=min_lr  
                        )

            # 第二阶段：使用余弦退火
            else:
                scheduler2.step()

            # 打印训练信息
            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, '
                      f'LR: {current_lr:.6f}, Stage: {stage}, '
                      f'Epochs since best: {epochs_since_best}')
                        
            # 如果连续patience轮都没有超过最佳损失，则早停
            if epochs_since_best >= patience:
                print(f'连续{epochs_since_best}轮未出现显著改善(改善阈值:{min_delta})，在第{epoch+1}轮提前停止训练')
                break

        # 使用最佳模型评估
        self.model = best_model
        with torch.no_grad():
            y_pred,_ = self.model(self.X_tensor)
            mse = criterion(y_pred, self.y_tensor).item()
            print(f"训练的MSE: {mse:.4f}")

        return mse


    def predict(self, X_new):
        """
        预测新数据
        
        参数:
            X_new: numpy array, 需要预测的新数据，需确保新的数据与训练数据具有相同的分布特征
        返回:
            预测值（原始尺度）
        """
        # 标准化输入数据
        X_new_scaled = (X_new - self.X_mean) / self.X_std
        X_new_tensor = torch.FloatTensor(X_new_scaled)
        
        # 预测
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():
            y_pred_scaled,_ = self.model(X_new_tensor)
            
        # 还原预测值到原始尺度
        y_pred = y_pred_scaled.numpy() * self.y_std + self.y_mean
        
        return y_pred
    



    def create_network(self, input_size=2, hidden_size=20, output_size=1):
        """创建神经网络，通过控制权重使折点均匀分布"""
        class Net(nn.Module):
            def __init__(self, X_cols, X_ranges_standardized):
                super(Net, self).__init__()
                self.hidden = nn.Linear(input_size, hidden_size)
                self.output = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.X_cols = X_cols
                self.X_ranges_standardized = X_ranges_standardized
                
                with torch.no_grad():
                    # 1. 随机生成偏置值（不为0）
                    biases = torch.randn(hidden_size)
                    biases[biases.abs() < 0.1] = 0.1
                    self.hidden.bias.data = biases
                    
                    # 2. 对每个输入变量计算合适的权重
                    for i, col in enumerate(self.X_cols):
                        # 获取该变量的范围
                        x_min, x_max = self.X_ranges_standardized[col]
                        
                        # 生成目标折点（随机分布）
                        desired_thresholds = x_min + (x_max - x_min) * torch.rand(hidden_size)
                        
                        # 根据目标折点和偏置计算权重
                        weights = -biases / desired_thresholds
                        
                        # 设置该输入变量的权重
                        self.hidden.weight.data[:, i] = weights
                    
                    # 3. 输出层使用kaiming初始化
                    nn.init.kaiming_uniform_(
                        self.output.weight, 
                        nonlinearity='relu'
                    )
            
            def forward(self, x):
                # 计算折点惩罚
                penalty = 0
                for i, col in enumerate(self.X_cols):
                    x_min, x_max = self.X_ranges_standardized[col]
                    range_width = x_max - x_min
                    
                    # 定义缓冲区（范围的10%）
                    buffer = range_width * 0.1
                    soft_min = x_min + buffer
                    soft_max = x_max - buffer
                    
                    # 计算当前折点
                    weights = self.hidden.weight[:, i]
                    thresholds = -self.hidden.bias / weights
                    
                    # 计算超出软边界的部分，并限制最大值
                    left_violation = torch.clamp(soft_min - thresholds, min=0, max=5)
                    right_violation = torch.clamp(thresholds - soft_max, min=0, max=5)
                    
                    # 使用更温和的惩罚函数
                    penalty += torch.mean(left_violation**2 + right_violation**2)
                    
                    # 或者使用有界的指数惩罚
                    # penalty += torch.mean(torch.exp(torch.clamp(left_violation, max=2)) + 
                    #                      torch.exp(torch.clamp(right_violation, max=2)) - 2)
                
                # 正常的前向传播
                x = self.relu(self.hidden(x))
                x = self.output(x)
                return x, penalty        
        self.model = Net(self.X_cols, self.X_ranges_standardized)
        
        # 打印初始化信息
        print("\n神经网络模型已创建:")
        print(f"- 输入层: {input_size}个神经元")
        print(f"- 隐藏层: {hidden_size}个神经元")
        print(f"- 输出层: {output_size}个神经元")
        
        # 验证每个变量的初始折点
        with torch.no_grad():
            for i, col in enumerate(self.X_cols):
                weights = self.model.hidden.weight.data[:, i]
                biases = self.model.hidden.bias.data
                thresholds = -biases / weights
                print(f"\n{col}的初始折点位置:")
                print(f"范围: [{self.X_ranges_standardized[col][0]:.4f}, "
                    f"{self.X_ranges_standardized[col][1]:.4f}]")
                print("折点:", sorted(thresholds.numpy()))
    def plot_mse_trend(self, mse_list):
        """绘制神经元数量与MSE的关系图"""
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.figure(figsize=(10, 6))
        neurons, avg_mses, min_mses, harmonic_mses = zip(*mse_list)  # 解压缩数据
        plt.plot(neurons, avg_mses, 'b-o', label='平均MSE')
        plt.plot(neurons, min_mses, 'g-o', label='最小MSE')
        plt.plot(neurons, harmonic_mses, 'y-o', label='调和平均MSE')
        plt.xlabel('神经元数量')
        plt.ylabel('MSE')
        plt.title('神经元数量与MSE的关系')
        plt.legend()
        plt.grid(True)
        plt.show()
    def plot_threshold_stability(self, threshold_stability_list=None, smooth=False):
        """绘制神经元数量与折点稳定性的关系图
        有必要进行平滑处理，因为n在增加的时候，相邻的两个n的折点可能会波动，
            比如二次函数，整体上折点稳定性会增加，但在相邻的两个n中，由于是对称图形，奇数个折点的稳定性会更强
            偶数个折点稳定性会减弱，在一个折点到两个折点的过程中，稳定性会大幅下降
        """
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建字典来存储每个变量的数据
        var_data = {}
        summary_data = {"root mean square": {"neurons": [], "stability": []},
                    "mean": {"neurons": [], "stability": []}}
        
        for n, var, stability in threshold_stability_list:
            if var in ["root mean square", "mean"]:
                summary_data[var]["neurons"].append(n)
                summary_data[var]["stability"].append(stability)
            else:
                if var not in var_data:
                    var_data[var] = {"neurons": [], "stability": []}
                var_data[var]["neurons"].append(n)
                var_data[var]["stability"].append(stability)
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # 第一个子图：各变量的折点稳定性
        colors = plt.cm.rainbow(np.linspace(0, 1, len(var_data)))
        for (var, data), color in zip(var_data.items(), colors):  # 这里是问题所在
            ax1.plot(data["neurons"], data["stability"], '-o', 
                    label=var, color=color, alpha=0.5 if smooth else 1)
            if smooth:
                smoothed_stability = [data["stability"][-1]]
                for i in range(len(data["stability"]) - 1, 0, -1):  # 修正了索引范围
                    smoothed_stability.append(data["stability"][i] * 0.25 + data["stability"][i-1] * 0.75)
                smoothed_stability = smoothed_stability[::-1]
                ax1.plot(data["neurons"], smoothed_stability, '-', color=color, alpha=1)  # 将虚线改为实线
        ax1.set_xlabel('神经元数量')
        ax1.set_ylabel('折点稳定性')
        ax1.set_title('各变量的折点稳定性')
        ax1.legend()
        ax1.grid(True)
        
        # 第二个子图：均方根和均值稳定性
        ax2.plot(summary_data["root mean square"]["neurons"], 
                summary_data["root mean square"]["stability"], 
                '-o', label='均方根稳定性', color='black', alpha=0.5 if smooth else 1)
        ax2.plot(summary_data["mean"]["neurons"],
                summary_data["mean"]["stability"],
                '-o', label='平均稳定性', color='red', alpha=0.5 if smooth else 1)
        if smooth:
            smoothed_root_mean_square = [summary_data["root mean square"]["stability"][-1]]
            smoothed_mean = [summary_data["mean"]["stability"][-1]]
            for i in range(len(summary_data["root mean square"]["stability"])-1,0,-1):
                smoothed_root_mean_square.append(summary_data["root mean square"]["stability"][i]*0.25 + summary_data["root mean square"]["stability"][i-1]*0.75)
                smoothed_mean.append(summary_data["mean"]["stability"][i]*0.25 + summary_data["mean"]["stability"][i-1]*0.75)
            smoothed_root_mean_square = smoothed_root_mean_square[::-1]
            smoothed_mean = smoothed_mean[::-1]
            ax2.plot(summary_data["root mean square"]["neurons"], smoothed_root_mean_square, '-', color='black', alpha=1)  # 将虚线改为实线
            ax2.plot(summary_data["mean"]["neurons"], smoothed_mean, '-', color='red', alpha=1)  # 将虚线改为实线
        ax2.set_xlabel('神经元数量')
        ax2.set_ylabel('折点稳定性')
        ax2.set_title('总体稳定性指标')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_mse_and_stability(self, analysis_results, mse_metrics, stability_metrics, max_n):
        """
        绘制MSE和稳定性指标的关系图
        
        参数:
            analysis_results: tuple, neuron_mse_analysis函数的返回值,包含(mse_list, threshold_stability_list)
            mse_metrics: list, 指定要绘制的MSE指标,可包含'avg','min','har'
            stability_metrics: list, 指定要绘制的稳定性指标,可包含变量名和'mean','root mean square'
            max_n: int, 指定绘制的最大神经元数量范围(1到max_n)
        """
        mse_list, threshold_stability_list = analysis_results
        
        # 设置中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图和两个y轴
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        # 处理MSE数据 - 使用左侧y轴(ax1)
        metric_names = {'avg': '平均MSE', 'min': '最小MSE', 'har': '调和平均MSE'}
        colors = plt.cm.rainbow(np.linspace(0, 1, len(mse_metrics)))
        
        for metric, color in zip(mse_metrics, colors):
            if metric == 'avg':
                data = [(n, avg) for n, avg, _, _ in mse_list 
                       if n <= max_n and 1 <= n]
            elif metric == 'min':
                data = [(n, min_mse) for n, _, min_mse, _ in mse_list 
                       if n <= max_n and 1 <= n]
            elif metric == 'har':
                data = [(n, har) for n, _, _, har in mse_list 
                       if n <= max_n and 1 <= n]
                
            n_values, metric_values = zip(*data)
            line1 = ax1.plot(n_values, metric_values, '-o', 
                    label=metric_names[metric], color=color)
            
        ax1.set_xlabel('神经元数量')
        ax1.set_ylabel('MSE', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # 处理稳定性数据 - 使用右侧y轴(ax2)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(stability_metrics)))
        
        for metric, color in zip(stability_metrics, colors):
            stability_data = [(n, stab) for n, var, stab in threshold_stability_list 
                            if var == metric and n <= max_n and 1 <= n]
            if stability_data:
                n_values, stab_values = zip(*stability_data)
                if metric in ['mean', 'root mean square']:
                    label = '平均稳定性' if metric == 'mean' else '均方根稳定性'
                    linewidth = 3
                else:
                    label = f'{metric}的稳定性'
                    linewidth = 1
                line2 = ax2.plot(n_values, stab_values, '--o', 
                        label=label, color=color, linewidth=linewidth)
        
        ax2.set_ylabel('折点稳定性', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # 合并两个轴的图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('神经元数量与MSE和稳定性的关系')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def NN_multifit(self, n, trials=50, epochs=5000, learning_rate=0.1, min_lr=0.01, switch_threshold=0.01, patience=1000, min_delta=1e-7, decay_rate=[0.8,50], penalty_strength=1, pop_ratio=0.5):
        """
        多次训练神经网络以获得稳定的结果。
        
        该函数会多次训练神经网络,并根据性能指标筛选出最优的训练结果。通过多次训练和筛选,
        可以降低单次训练的随机性带来的影响,得到更稳定可靠的模型。
        
        参数:
            n: int, 隐藏层神经元数量
            trials: int, 训练总次数,默认为50次
            epochs: int, 每次训练的迭代次数,默认5000
            learning_rate: float, 初始学习率,默认0.1
            min_lr: float, 最小学习率,默认0.01
            switch_threshold: float, 切换学习率策略的阈值,默认0.01
            patience: int, 早停的耐心值,默认1000
            min_delta: float, 判定损失改善的最小阈值,默认1e-7
            decay_rate: list, 学习率衰减参数[factor, patience],默认[0.8,50]
            penalty_strength: float, 惩罚项强度,默认1
            pop_ratio: float, 要剔除的较差训练结果的比例,默认0.5
            
        返回:
            mse_list: list, 所有保留训练结果的MSE值列表
            best_models_list: list, 保留的最佳模型列表
            n: int, 隐藏层神经元数量
        """
        import copy
        
        mse_list = []  # 存储每次训练的MSE
        best_models_list = []  # 存储每次训练的最佳模型
        #param_cov_list = []  # 存储每次训练的参数方差-协方差矩阵
        
        # 设置批量大小
        batch_size = min(512, len(self.X))
        total_batches = len(self.X) // batch_size + (1 if len(self.X) % batch_size != 0 else 0)
        
        trial = 0
        while trial < trials:
            print(f"\n第 {trial+1}/{trials} 次训练")
            print(f"神经元数量: {n}")
            print(f"学习率: {learning_rate}")
            
            # 创建模型
            input_size = len(self.X_cols)
            self.create_network(input_size=input_size, hidden_size=n, output_size=1)
            
            # 准备数据批次
            perm = torch.randperm(len(self.X_tensor))
            batch_indices = [(i*batch_size, min((i+1)*batch_size, len(self.X))) for i in range(total_batches)]
            X_batches = [self.X_tensor[perm[start:end]] for start, end in batch_indices]
            y_batches = [self.y_tensor[perm[start:end]] for start, end in batch_indices]
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=decay_rate[0], 
                patience=decay_rate[1], verbose=False, min_lr=min_lr
            )
            
            # 训练过程变量初始化
            recent_losses = []
            stage = 1
            best_model = None
            best_loss = float('inf')
            epochs_since_best = 0
            need_retrain = False
            
            # 用于收集梯度的列表
            #gradients = []
            
            # 训练模型
            for epoch in range(epochs):
                epoch_loss = 0.0


                for b in range(total_batches):
                    optimizer.zero_grad()
                    outputs, penalty = self.model(X_batches[b])
                    loss = criterion(outputs, y_batches[b]) + penalty_strength * penalty
                    
                    loss.backward()

                    
                    # 收集梯度
                    #current_gradients = []
                    #for param in self.model.parameters():
                    #    if param.grad is not None:
                    #        current_gradients.append(param.grad.view(-1))
                    #gradients.append(torch.cat(current_gradients))
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    epoch_loss += loss.item()

                
                epoch_loss /= total_batches

                recent_losses.append(epoch_loss)
                

                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f},  epochs since best: {epochs_since_best}")

                # 第100轮loss检查
                if epoch == 99 and epoch_loss > 10:
                    need_retrain = True
                    print(f"第100轮loss为{epoch_loss:.4f}，大于10，需要重新训练")
                    break
                
                # 更新最佳模型
                if epoch_loss < (best_loss - min_delta):
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(self.model)
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                
                # 学习率调度
                if stage == 1:
                    scheduler1.step(epoch_loss)
                    if len(recent_losses) >= 107:
                        recent_change = abs(np.mean(recent_losses[-6:-1]) - np.mean(recent_losses[-106:-101])) / np.mean(recent_losses[-106:-101])
                        if recent_change < switch_threshold:
                            stage = 2
                            print(f"切换到第二阶段，当前epoch: {epoch+1}")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=epoch, eta_min=min_lr
                            )
                else:
                    scheduler2.step()
                
                # 早停
                if epochs_since_best >= patience:
                    print(f"早停触发，{patience}轮未改善")
                    break
            
            if need_retrain:
                continue
                
           # 使用最佳模型进行评估
            self.model = best_model

            # 计算参数的方差-协方差矩阵
            #gradients = torch.stack(gradients).cpu()  # 将梯度移至CPU
            #mean_gradients = gradients.mean(dim=0, keepdim=True)  # 计算梯度均值
            #centered_gradients = gradients - mean_gradients  # 中心化梯度
            #param_cov = torch.mm(centered_gradients.t(), centered_gradients) / (len(gradients) - 1)  # 无偏协方差矩阵

            # 将最佳模型和方差-协方差矩阵添加到列表中
            #param_cov_list.append(param_cov)

            # 将最佳模型添加到列表中
            best_models_list.append(copy.deepcopy(best_model))
            with torch.no_grad():
                y_pred, _ = self.model(self.X_tensor)
                mse = criterion(y_pred, self.y_tensor).item()
                print(f"最终MSE: {mse:.6f}")
                mse_list.append(mse)
                

            
            trial += 1
            
        # 根据MSE筛选最好的trials
        sorted_indices = np.argsort(mse_list)  # 按MSE排序
        keep_trials = max(1, int(trials * (1-pop_ratio)))  # 保留的trials数量,至少保留1个
        kept_indices = sorted_indices[:keep_trials]  # 保留的trial索引
        print(f"\n保留表现最好的 {keep_trials} 次训练结果")
        


        best_models_list = [best_models_list[i] for i in kept_indices]
            
        #param_cov_list = [param_cov_list[i] for i in kept_indices]
        
        mse_list = [mse_list[i] for i in kept_indices]


        
        return (mse_list,  best_models_list,n)
    def plot_average_effects(self, results, x_name,k=5000,caculate_significance = False, train_control_model_setting=[5,1000,0.1,[0.9,50],0.5],bboxtoanchor2=(0.5,0.8),bboxtoanchor3=(0.5,0.7)):
        """
        为指定变量绘制加权预测曲线,并可选择性地计算统计显著性。

        该函数通过对多个神经网络模型的预测结果进行加权平均,生成一条平滑的预测曲线,用于展示自变量与因变量之间的关系。
        同时支持计算置信区间和统计显著性检验。

        参数:
            results: tuple, 包含训练结果的元组,具体包括:
                - mse_list: list, 每个模型的均方误差
                - best_models_list: list, 训练好的神经网络模型列表
                - n: int, 训练样本数量
            x_name: str, 要分析的自变量名称
            k: int, default=5000, 用于绘图的采样点数量
            caculate_significance: bool, default=False, 是否计算统计显著性
            train_control_model_setting: list, default=[5,1000,0.1,[0.9,50],0.5], 控制模型的训练参数设置:
                - [0]: int, 训练次数
                - [1]: int, 每次训练的最大迭代次数
                - [2]: float, 学习率
                - [3]: list, [早停参数, 早停轮数]
                - [4]: float, 保留比例
            bboxtoanchor2: tuple, default=(1.04,1), 图例2的位置参数
            bboxtoanchor3: tuple, default=(1.04,1), 图例3的位置参数

        返回:
            无直接返回值,但会生成以下可视化结果:
            1. 加权预测曲线
            2. 95%置信区间(如果样本量足够)
            3. 统计显著性检验结果(如果caculate_significance=True)

        注意:
            - 该函数依赖于seaborn库进行可视化
            - 计算统计显著性时会消耗较多计算资源
            - 建议在样本量充足时使用置信区间功能
        """
        import seaborn as sns
        
            
        mse_list,  best_models_list,n = results
        # 获取x_name对应的列索引
        x_idx = self.X_cols.index(x_name)
        
        # 找到x_name变量的最大值和最小值的索引
        x_values = self.X_tensor[:, x_idx]
        min_idx = torch.argmin(x_values).item()
        max_idx = torch.argmax(x_values).item()
        # 随机抽取k个点（无放回）
        n_samples = min(k-2, len(self.X_tensor)-2)
        mask = torch.ones(len(self.X_tensor), dtype=torch.bool)
        mask[min_idx] = False
        mask[max_idx] = False
        available_indices = torch.arange(len(self.X_tensor))[mask]
        random_indices = np.random.choice(available_indices.numpy(), size=n_samples, replace=False)
        
        # 合并最大值、最小值点和随机点
        all_indices = np.concatenate([[min_idx, max_idx], random_indices])
        sampled_points = self.X_tensor[all_indices]
        sampled_y = self.y_tensor[all_indices]  # 记录对应的y值
        # 获取x_name对应的列索引
        x_idx = self.X_cols.index(x_name)
        
        # 根据x_name对应的列进行排序
        sort_indices = torch.argsort(sampled_points[:, x_idx])
        sampled_points = sampled_points[sort_indices]
        sampled_y = sampled_y[sort_indices]  # y值也按照相同顺序排序
        
        # 存储所有抽样点和对应预测结果
        sampled_predictions = []  # 存储每个模型的预测结果
        
        # 对每个保留的模型进行预测
        for model in best_models_list:
            # 使用当前模型进行预测
            with torch.no_grad():
                predictions = model(sampled_points)[0].detach().numpy()

            sampled_predictions.append(predictions)
            
        # 保存抽样点
        sampled_x = sampled_points[:, self.X_cols.index(x_name)].numpy()

        
        # 检查是否有数据
        if len(mse_list) == 0:
            print("没有可用的训练结果")
            return
                
        # 计算每个trial的权重(使用MSE的倒数)
        weights = [1/mse for mse in mse_list]
        weights = np.array(weights) / sum(weights)  # 归一化权重
        
        # 获取该变量的所有预测结果
        x_predictions = sampled_predictions
        
        # 获取x的标准化范围
        if x_name not in self.X_ranges_standardized:
            print(f"变量 {x_name} 不在标准化范围中")
            return
            

        
        # 获取原始x数据点
        try:
            x_idx = self.X_cols.index(x_name)
        except ValueError:
            print(f"变量 {x_name} 不在X_cols中")
            return
            
        x_data = self.X_tensor[:, x_idx].numpy()
        x_data_original = x_data * self.X_std[x_idx] + self.X_mean[x_idx]
        y_data_original = self.y_tensor.numpy() * self.y_std + self.y_mean
        
        
        
        # 计算曲线的加权预测值（用于绘制平滑曲线）
        weighted_curve = np.zeros(k)
        for predictions, weight in zip(x_predictions, weights):
            weighted_curve += predictions.flatten() * weight
        x_original = sampled_x * self.X_std[x_idx] + self.X_mean[x_idx]
        y_original = sampled_y * self.y_std + self.y_mean
        y_net = weighted_curve * self.y_std + self.y_mean
        

        
        # 使用Savitzky-Golay滤波器平滑曲线
        from scipy.signal import savgol_filter
        import statsmodels.api as sm
        # 使用LOWESS得到平滑曲线
        y_curve_raw = sm.nonparametric.lowess(y_net, x_original, frac=0.1, it=3)[:, 1]
        
        # 创建插值函数
        from scipy.interpolate import interp1d
        f = interp1d(x_original, y_curve_raw)
        
        # 在新的x点上计算插值
        x_new = np.linspace(min(x_original), max(x_original), 100)
        y_curve = f(x_new)

        
        # 计算所有模型在所有采样点上的折点
        all_thresholds = []
        for model in best_models_list:
            # 对每个模型计算所有采样点的折点
            thresholds_list, thresholds_original_list = self.calculate_conditional_thresholds_batch(
                model, 
                x_name,
                sampled_points.numpy()
            )

            # 将每个模型的所有折点添加到总列表中
            all_thresholds.append(np.concatenate(thresholds_original_list))

            
        # 使用标准化后的范围进行筛选
        x_min_stan = np.min(x_data)
        x_max_stan = np.max(x_data)


        
        # 剔除每个模型中超出范围的折点，并从每个模型中抽取1000个折点
        filtered_thresholds = []
        for thresholds in all_thresholds:

            # 计算5%和95%分位点
            lower_bound = np.percentile(thresholds, 5)
            upper_bound = np.percentile(thresholds, 95)
            valid_thresholds = thresholds[(thresholds >= lower_bound) & (thresholds <= upper_bound)]


            if len(valid_thresholds) >= 1000:
                sampled_thresholds = np.random.choice(valid_thresholds, size=1000, replace=False)
            else:
                sampled_thresholds = valid_thresholds
            filtered_thresholds.append(sampled_thresholds)

        
        # 创建子图
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 计算y轴范围
        y_quantiles = np.quantile(self.y, [0.025, 0.975])  # 计算y的2.5%和97.5%分位数
        y_min = y_quantiles[0]
        y_max = y_quantiles[1]
        # 计算x轴范围
        x_range = max(x_original) - min(x_original)

        # 在下面的子图创建双y轴
        ax2_density = ax2.twinx()

        ax2.set_xlabel(f'{x_name}')
        ax2.set_ylabel(self.y_col)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xlim(min(x_original), max(x_original))  # 设置x轴范围
        
        # 在右轴绘制折点的密度分布
        if len(filtered_thresholds) > 0:
            # 使用核密度估计绘制平滑的直方图
            _,density_values = self.plot_weighted_density(weights, filtered_thresholds,np.linspace(min(x_original), max(x_original), 100), ax=ax2_density, color='orange')
            ax2_density.set_ylabel('弯折概率密度')
            density_values = density_values * x_range/100

            # 设置密度图y轴上限为10/(xmax-xmin)
            ax2_density.set_ylim(0, 10/x_range)
            ax2_density.set_xlim(min(x_original), max(x_original))  # 设置x轴范围
        


        y_curve_smooth = savgol_filter(y_curve, window_length=100, polyorder=3)

        # 计算角度变化的绝对值（弧度）
        # 计算每一段与y轴正方向的逆时针夹角（弧度）
        angles = []
        DX = []
        DY = []

        for i in range(len(y_curve_smooth)-1):
            dx = x_range/99
            dy = y_curve_smooth[i+1] - y_curve_smooth[i]
            DX.append(dx)
            DY.append(dy)
        # 计算DX和DY的平方和除以100
        dx_squared_sum = np.sqrt(sum([dx**2 for dx in DX])/99)
        DX = [dx/dx_squared_sum for dx in DX]
        dy_squared_sum = np.sqrt(sum([dy**2 for dy in DY])/99)
        DY = [dy/dy_squared_sum for dy in DY]
        for dx,dy in zip(DX,DY):
            # 计算与y轴正方向的逆时针夹角
            angle = np.arctan2(dy, dx)+np.pi/2
            

            
            angles.append(angle)

        # 计算角度变化（弧度）
        angle_changes = []
        for i in range(len(angles)-1):
            # 直接计算逆时针方向的角度变化
            angle_change = angles[i+1] - angles[i]
            
                
            angle_changes.append(angle_change)

        # 将angle_changes替换原来的slope_changes
        slope_changes = [abs(angle) for angle in angle_changes]


        



        # 将斜率变化率与密度值相乘 对于弯折程度的衡量，一是要看在此处弯折的概率，二是要看在此处弯折的幅度
        weighted_slope_changes = []
        for i in range(len(slope_changes)):
            weighted_slope_changes.append(slope_changes[i] * density_values[i])
        








        significant = []
        for i in range(len(weighted_slope_changes)):
            if weighted_slope_changes[i] > 0.0001:
                significant.append(True)
            else:
                significant.append(False)
        for i in range(len(significant)):
            if significant[i]:
                if i > 0 and i < len(significant)-1:
                    if (angle_changes[i-1]*angle_changes[i] < 0) or (angle_changes[i+1]*angle_changes[i] < 0):
                        significant[i] = False
        
        breakpoints = []


        centerlist = []
        centerweight = []
        for i in range(len(significant)):
            if i > 0 and i < len(significant)-1:
                if (significant[i] and significant[i-1]):
                    centerlist.append(i)
                    centerweight.append(weighted_slope_changes[i])
                    if (not significant[i+1]) or (i == len(significant)-2):
                        weighted_center = sum(c * w for c, w in zip(centerlist, centerweight)) / sum(centerweight)
                        breakpoints.append(weighted_center)
                        centerlist = []
                        centerweight = []
        


        dx = (x_max_stan - x_min_stan)/99
        breakpoints = [x_min_stan + i*dx for i in breakpoints]
        breakpoints.append(x_max_stan)
        breakpoints.append(x_min_stan)
        breakpoints = sorted(breakpoints)
        
        criterion = nn.MSELoss()
        if caculate_significance == True:

            likelihood_ratios = []
            # 遍历breakpoints中相邻的两个点,作为区间的起点和终点
            for i in range(len(breakpoints)-1):
                rangemin = breakpoints[i]
                rangemax = breakpoints[i+1]
                #nn = int(((rangemax - rangemin)/(breakpoints[-1] - breakpoints[0]))*n)
                best_controlmodels,_ = self.train_control_model(n,
                                                        trials=train_control_model_setting[0],
                                                        epochs=train_control_model_setting[1], 
                                                        special_index=x_idx,
                                                        range_min=rangemin,
                                                        range_max=rangemax,
                                                        learning_rate=train_control_model_setting[2],
                                                        decay_rate=train_control_model_setting[3],
                                                        pop_ratio=train_control_model_setting[4])
                
                # 筛选出在rangemin和rangemax之间的点
                mask = (x_data >= rangemin) & (x_data <= rangemax)
                x_data_in_range = self.X_tensor[mask]
                y_data_in_range = self.y_tensor[mask]
                with torch.no_grad():
                    mse_list_control = [criterion(model(x_data_in_range)[0], y_data_in_range).item() for model in best_controlmodels]
                with torch.no_grad():
                    experiment_mse_list = [criterion(model(x_data_in_range)[0], y_data_in_range).item() for model in best_models_list]


               
                N = len(x_data_in_range)
                
                # 计算控制模型的RSS
                if len(x_data_in_range) > 0:
                    weights_control = [1/mse**2 for mse in mse_list_control]
                    weights_control = np.array(weights_control) / sum(weights_control)  # 归一化权重
                    control_sigma2 = np.sum([mse * weight for mse, weight in zip(mse_list_control, weights_control)])
                    
                    '''with torch.no_grad():
                        control_y_preds = [model(x_data_in_range)[0].numpy() for model in best_controlmodels]
                        residuals_control = [y_data_in_range - pred for pred in control_y_preds]
                        weighted_residuals_control = np.sum([res * weight for res, weight in zip(residuals_control, weights_control)], axis=0)

                    
                    # 计算控制模型的RSS
                    control_rss = np.sum((weighted_residuals_control) ** 2)
                    control_sigma2 = control_rss / N'''
                    



                    weights_main = [1/mse**2 for mse in experiment_mse_list]
                    weights_main = np.array(weights_main) / sum(weights_main)  # 归一化权重
                    experiment_sigma2 = np.sum([mse * weight for mse, weight in zip(experiment_mse_list, weights_main)])
                    ''' with torch.no_grad():
                        main_y_preds = [model(x_data_in_range)[0].numpy() for model in best_models_list]
                        residuals_main = [y_data_in_range - pred for pred in main_y_preds]
                        weighted_residuals_main = np.sum([res * weight for res, weight in zip(residuals_main, weights_main)], axis=0)

                    # 计算实验模型的RSS和sigma2
                    experiment_rss = np.sum((weighted_residuals_main) ** 2)
                    experiment_sigma2 = experiment_rss / N'''
                    
                    
                    likelihood_ratio = N*np.log(control_sigma2/experiment_sigma2)
                    if likelihood_ratio < 0:
                        print("likelihood_ratio≈0，可增大训练次数或神经元数量试试")
                        likelihood_ratio = 0
                    likelihood_ratios.append(likelihood_ratio)

            # 将breakpoints还原为原始空间的数据
            breakpoints_original = [bp * self.X_std[x_idx] + self.X_mean[x_idx] for bp in breakpoints]
            breakpoints = breakpoints_original
            
            print("breakpoints:", breakpoints)
            print("likelihood_ratios:", likelihood_ratios)
                    

            # 绘制第三张图
            # 根据breakpoints划分区间
            x_points = np.linspace(min(x_original), max(x_original), 100)
            y_points = y_curve_smooth
            
            # 为每个区间选择颜色
            for i in range(len(breakpoints)-1):
                mask = (x_points >= breakpoints[i]) & (x_points <= (breakpoints[i+1]+2*dx*self.X_std[x_idx]))
                if likelihood_ratios[i] > 10.828:
                    color = 'green'
                elif likelihood_ratios[i] > 6.635:
                    color = 'yellowgreen' 
                elif likelihood_ratios[i] > 3.841:
                    color = 'yellow'
                elif likelihood_ratios[i] > 2.706:
                    color = '#D4D462'  # 灰黄色
                else:
                    color = 'gray'
                
                ax3.plot(x_points[mask], y_points[mask], '-', color=color, linewidth=2)
                

            # 计算每一段的斜率
            slopes = []
            for i in range(len(x_points)-1):
                dx = x_points[i+1] - x_points[i]
                dy = y_points[i+1] - y_points[i] 
                slope = dy/dx
                slopes.append(slope)
            slopes_x = [x_points[i] + dx/2 for i in range(len(x_points)-1)]


            weighted_slope_changes_copy = weighted_slope_changes
            weighted_slope_changes_copy = [weighted_slope_changes_copy[0]] + weighted_slope_changes_copy + [weighted_slope_changes_copy[-1]]
            # 计算weighted_slope_changes_copy相邻元素的几何均值
            geometric_means = []
            for i in range(len(weighted_slope_changes_copy)-1):
                geometric_mean = np.sqrt(weighted_slope_changes_copy[i] * weighted_slope_changes_copy[i+1])
                geometric_means.append(geometric_mean)
            

            main_AMEs = []
            # 对每个区间进行处理
            for i in range(len(breakpoints)-1):
                # 筛选当前区间内的点
                mask = (np.array(slopes_x) >= breakpoints[i]) & (np.array(slopes_x) <= breakpoints[i+1])
                # 获取当前区间的斜率
                interval_slopes = np.array(slopes)[mask]
                # 获取当前区间的geometric_means
                interval_geometric_means = np.array(geometric_means)[mask]
                # 计算权重(interval_geometric_means的倒数)
                weights1 = 1 / interval_geometric_means
                # 计算加权平均值
                weighted_avg = np.average(interval_slopes, weights=weights1)
                main_AMEs.append(weighted_avg)
                

            # 设置文本y坐标位置(y_max下方10%)
            text_y = y_max - (y_max - y_min) * 0.05
            
            # 在每个区间中心添加文本
            for i in range(len(breakpoints)-1):
                # 计算区间中心点x坐标
                center_x = (breakpoints[i] + breakpoints[i+1]) / 2
                
                # 根据不同的likelihood ratio显示不同的显著性标记
                if likelihood_ratios[i] > 10.828:
                    sig = 'LRT ≈ ' + str(round(likelihood_ratios[i], 2)) + '\n' + 'AME ≈ ' + str(round(main_AMEs[i], 2))
                    color = 'green' 
                elif likelihood_ratios[i] > 6.635:
                    sig = 'LRT ≈ ' + str(round(likelihood_ratios[i], 2)) + '\n' + 'AME ≈ ' + str(round(main_AMEs[i], 2))
                    color = 'yellowgreen'
                elif likelihood_ratios[i] > 3.841:
                    sig = 'LRT ≈ ' + str(round(likelihood_ratios[i], 2)) + '\n' + 'AME ≈ ' + str(round(main_AMEs[i], 2))
                    color = 'yellow'
                elif likelihood_ratios[i] > 2.706:
                    sig = 'LRT ≈ ' + str(round(likelihood_ratios[i], 2)) + '\n' + 'AME ≈ ' + str(round(main_AMEs[i], 2))
                    color = '#D4D462'
                else:
                    sig = 'LRT ≈ ' + str(round(likelihood_ratios[i], 2)) + '\n' + 'AME ≈ ' + str(round(main_AMEs[i], 2))
                    color = 'gray'
                    
                # 添加文本
                ax3.text(center_x, text_y, sig, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=10,
                        color=color,
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=3))
            for bp in breakpoints:
                ax3.axvline(x=bp, color='gray', linestyle='--', alpha=0.5)
                
        else:
            x_points = np.linspace(min(x_original), max(x_original), 100)
            y_points = y_curve_smooth
            ax3.plot(x_points, y_points, '-', color='red', linewidth=2)


        # 计算residual并加上预测曲线
        residuals = np.array(y_original).flatten() - np.array(y_net).flatten()
        y_with_residuals = np.add(y_curve_raw, residuals)
        ax3.scatter(x_original, y_with_residuals, c='black', s=0.5, alpha=0.2)

        ax3.set_xlabel(x_name)
        ax3.set_ylabel(self.y_col)
        ax3.set_title(f'{self.y_col}对{x_name}的平均效应')
        ax3.set_ylim(y_min, y_max)
        ax3.set_xlim(min(x_original), max(x_original))
        
        # 添加图例
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        legend_elements = [
            mlines.Line2D([], [], color='black', marker='.', linestyle='None', markersize=3, label=f'{self.y_col}对{x_name}的平均效应+残差', alpha=0.4),
            mlines.Line2D([], [], color='grey', linestyle='-', linewidth=1.2, label='                                    残差走势'),
            mpatches.Patch(color='green',       label='LRT > 10.828              P < 0.001'),
            mpatches.Patch(color='yellowgreen', label='6.635 < LRT ≤ 10.82   P < 0.01'),
            mpatches.Patch(color='yellow',      label='3.841 < LRT ≤ 6.635   P < 0.05'),
            mpatches.Patch(color='#D4D462',     label='2.706 < LRT ≤ 3.841   P < 0.10'),
            mpatches.Patch(color='gray',        label='LRT ≤ 2.706                P > 0.10')
        ]
        ax3.legend(handles=legend_elements, fontsize=7, bbox_to_anchor=bboxtoanchor3)






        # 先绘制热力图
        self.plot_heatmap_region(x_original, y_net, ax1)


        x_min_original = min(x_original)
        x_max_original = max(x_original)
        dx = (x_max_original - x_min_original)/99
        percents = [3,4] + [5]*95 +[4,3]
        percents = [p/99 for p in percents]
        # 初始化列表存储每个区间的数据点数量
        data_percents = []
        total_residuals_std = np.std(residuals)
        residuals_std = []
        
        # 遍历100个区间
        for i in range(99):
            # 计算当前区间的左右边界
            left_bound = (i-2)*dx + x_min_original
            right_bound = (i+3)*dx + x_min_original
            mask = (x_original >= left_bound) & (x_original <= right_bound)
            # 统计落在区间内的数据点数量
            count = np.sum(mask)
            residuals_std.append(np.std(residuals[mask])/total_residuals_std)
            
            data_percents.append(count/len(x_original))

        residuals_std = residuals_std + [residuals_std[-1]]
        # 使用高斯平滑处理residuals_std

        from scipy.ndimage import gaussian_filter1d
        residuals_std = gaussian_filter1d(residuals_std, sigma=6)


        ax3_twin = ax3.twinx()  # 创建双轴
        ax3_twin.plot(x_points, residuals_std, '-', color='grey', linewidth=1.2)
        print(x_points)
        print(residuals_std)
        ax3_twin.fill_between(x_points, residuals_std, 0, color='grey', alpha=0.4)
        ax3_twin.set_ylabel('局部残差/总体残差', color='black')
        ax3_twin.tick_params(axis='y', labelcolor='black')
        ax3_twin.set_ylim(0, 8)
        ax3_twin.set_yticks([1, 2])



        # 将data_percents除以percents
        data_percents = [dp/p for dp, p in zip(data_percents, percents)]

        
        data_percents_smooth = gaussian_filter1d(data_percents, sigma=2)
        data_percents_smooth[data_percents_smooth > 2] = 2
        data_percents_smooth[data_percents_smooth < 0.2] = 0.2
        data_percents_smooth =[i*1.5 for i in data_percents_smooth]

        
        
        
        # 在上面的子图绘制预测曲线
        points = np.array([np.linspace(min(x_original), max(x_original), 100), y_curve_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc1 = LineCollection(segments, color='#FF0000', linewidths=data_percents_smooth, label=f'{self.y_col}对{x_name}的平均效应')
        ax1.add_collection(lc1)
        ax1.scatter(x_original, y_original, c='black', s=0.5, alpha=0.4, label='样本点')
        ax1.set_xlabel(x_name)
        ax1.set_ylabel(self.y_col)
        ax1.set_title(f'{self.y_col}对{x_name}的平均效应')
        ax1.set_ylim(y_min, y_max)
        ax1.set_xlim(min(x_original), max(x_original))  # 设置x轴范围

        ax1.legend()
        

        
        # 设置固定的颜色映射范围
        min_threshold = 0.0001  # 可以调整这个值，低于这个值的都显示为蓝色
        max_threshold = 0.0003  # 可以调整这个值，高于这个值的都显示为红色
        norm = plt.Normalize(min_threshold, max_threshold)



        # 连续线条
        points = np.array([np.linspace(min(x_original), max(x_original), 100)[1:-1], y_curve_smooth[1:-1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap='coolwarm', norm=norm)
        lc.set_array(weighted_slope_changes[:-1])
        ax2.add_collection(lc)

        from matplotlib.cm import ScalarMappable
        # 添加自定义渐变色图例到 ax2 内部
        cax = ax2.inset_axes([0.2, 0.94, 0.6, 0.02])  # 调整到 ax2 内部顶部
        cbar = fig.colorbar(ScalarMappable(cmap='coolwarm', norm=norm), cax=cax, orientation='horizontal')
        cbar.set_ticks([])  # 去掉刻度和数值

        # 在图例两端添加文本
        cax.text(-0.05, 0.5, "弯折度低", va='center', ha='right', transform=cax.transAxes, fontsize=9)
        cax.text(1.05, 0.5, "弯折度高", va='center', ha='left', transform=cax.transAxes, fontsize=9)

        # 添加自动位置调整的图例，禁用顶部位置
        ax2.legend([plt.Line2D([0], [0], color='orange', lw=2)], ["弯折概率密度曲线"], loc="best", bbox_to_anchor=bboxtoanchor2, frameon=True, facecolor='white', edgecolor='gray', fontsize=10)




        plt.tight_layout()
        plt.show()
        

    def plot_heatmap_region(self, x_original, y_net, ax):
        """
        绘制单色热力图，使用透明度表示密度
        """
        # 创建网格
        x_grid = np.linspace(min(x_original), max(x_original), 100)
        y_grid = np.linspace(min(y_net), max(y_net), 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        from scipy.stats import gaussian_kde
        # 计算每个网格点的密度
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x_original, y_net])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        
        # 归一化密度值到0-1之间
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        
        # 设置透明度阈值和映射
        threshold = 0.2  # 低于此值的密度将完全透明
        max_alpha = 0.7  # 最大透明度
        
        # 创建透明度映射
        alphas = np.zeros_like(Z)
        mask = Z > threshold
        alphas[mask] = np.interp(Z[mask], 
                                [threshold, 1], 
                                [0, max_alpha])
        
        # 绘制热力图
        ax.imshow(alphas, 
                extent=[min(x_original), max(x_original), 
                        min(y_net), max(y_net)],
                origin='lower',
                aspect='auto',
                cmap='Blues',  # 使用蓝色色图
                alpha=1)  # 这里设置1是因为我们已经在alphas中设置了透明度

    def calculate_conditional_thresholds_batch(self, model, x_name, points):
        """
        批量计算在多个条件点下指定变量的折点位置
    
        参数：
            model: 神经网络模型
            x_name: str, 要计算折点的变量名称
            points: numpy.ndarray, shape=(n_points, n_features), 条件点矩阵
                    每行代表一个条件点，列的顺序需要与X_cols一致
        返回：
            thresholds_list: list of numpy.ndarray, 每个条件点下的折点位置列表（标准化空间）
            thresholds_original_list: list of numpy.ndarray, 每个条件点下的折点位置列表（原始空间）
        """
        # 获取x_name的索引
        x_idx = self.X_cols.index(x_name)
       
        # 检查输入维度
        if points.shape[1] != len(self.X_cols):
            raise ValueError(f"输入点的维度({points.shape[1]})与特征数量({len(self.X_cols)})不匹配")

        # 标准化输入点
        points_std = (points - self.X_mean) / self.X_std
    
        # 获取权重和偏置
        hidden_weights = model.hidden.weight.data.numpy()  # shape: (n_hidden, n_features)
        hidden_bias = model.hidden.bias.data.numpy()      # shape: (n_hidden,)
    
        # 创建掩码，去除x_name对应的列
        mask = np.ones(len(self.X_cols), dtype=bool)
        mask[x_idx] = False
        
        # 提取其他变量的权重和输入点
        other_weights = hidden_weights[:, mask]             # shape: (n_hidden, n_features-1)
        other_points = points_std[:, mask]                # shape: (n_points, n_features-1)
    
        # 计算其他变量的贡献 (矩阵乘法)
        # (n_points, n_features-1) @ (n_features-1, n_hidden) -> (n_points, n_hidden)
        other_vars_contribution = other_points @ other_weights.T
        
        # 获取目标变量的权重
        target_weights = hidden_weights[:, x_idx]         # shape: (n_hidden,)
       
        # 为每个点计算折点
        thresholds_list = []
        thresholds_original_list = []
        for i in range(len(points)):
            # 计算当前点的折点
            thresholds = -(hidden_bias + other_vars_contribution[i]) / target_weights
           
            # 转换回原始空间
            thresholds_original = thresholds * self.X_std[x_idx] + self.X_mean[x_idx]
           
            # 过滤无效值
            valid_mask = np.isfinite(thresholds_original)
            thresholds = thresholds[valid_mask]
            thresholds_original = thresholds_original[valid_mask]
    
            thresholds_list.append(thresholds)
            thresholds_original_list.append(thresholds_original)
        
        return thresholds_list, thresholds_original_list




    def plot_weighted_density(self, weights, breakpoints, x_values, ax=None, label='加权折点密度', color='blue', bw_method=0.4):
        """
        绘制加权概率密度图。

        参数：
            weights (list or numpy.ndarray): 每个模型的权重，长度应与breakpoints的长度相同。
            breakpoints (list of lists or numpy.ndarray): 每个模型的折点数据列表。
                                                    每个元素是一个包含该模型所有折点的列表或数组。
            x_values (numpy.ndarray): 要预测的x值
            ax (matplotlib.axes.Axes, optional): 要绘制的Axes对象。如果未提供，将创建一个新的图形。
            label (str, optional): 图例标签。
            color (str, optional): 密度曲线的颜色。
            bw_method (float, optional): 核密度估计的带宽方法。默认为1.0。
        """
        from scipy import stats
        import seaborn as sns
        if len(weights) != len(breakpoints):
            raise ValueError("weights的长度必须与breakpoints的长度相同。")

        # 扁平化所有折点数据和对应的权重
        all_breakpoints = []
        all_weights = []
        for weight, bp in zip(weights, breakpoints):
            all_breakpoints.extend(bp)
            all_weights.extend([weight] * len(bp))

        all_breakpoints = np.array(all_breakpoints)
        all_weights = np.array(all_weights)

        # 标准化权重，使其和为1
        all_weights = all_weights / np.sum(all_weights)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # 创建KDE对象，使用更大的带宽
        kde = gaussian_kde(all_breakpoints, weights=all_weights, bw_method=bw_method)
        
        # 计算密度值
        density_values = kde(x_values)
        
        # 归一化密度值
        density_values = density_values / np.trapz(density_values, x_values)
        
        # 绘制密度曲线
        ax.plot(x_values, density_values, color=color)
        ax.fill_between(x_values, 0, density_values, alpha=0.3, color=color)
        ax.set_xlabel('折点位置')
        ax.set_ylabel('密度')
        ax.set_title('加权折点概率密度分布')

        
        return x_values, density_values






    def create_control_network(self,input_size, hidden_size, output_size, special_index, range_min, range_max):
        """
        创建自定义神经网络的辅助函数
        
        参数:
            input_size (int): 输入特征的数量
            hidden_size (int): 隐藏层神经元数量
            output_size (int): 输出维度
            special_index (int): 特殊处理变量的索引
            range_min (float): 特殊区间的下界
            range_max (float): 特殊区间的上界
        
        返回:
            CustomNetwork: 创建的神经网络实例
        """


        class CustomNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, special_index, range_min, range_max, k=50):
                super().__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.special_index = special_index
                self.range_min = range_min
                self.range_max = range_max
                self.k = k
                self.hidden = nn.Linear(input_size, hidden_size)
                self.output = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()

                # 创建可训练参数
                # 对于每个神经元，创建一组权重和一个偏置
                self.weights = nn.Parameter(torch.randn(hidden_size, input_size))
                self.biases = nn.Parameter(torch.randn(hidden_size))
                self.p_values = nn.Parameter(torch.randn(hidden_size))  # 每个神经元的p值
                
                # 输出层参数
                self.output_weights = nn.Parameter(torch.randn(output_size, hidden_size))
                self.output_bias = nn.Parameter(torch.randn(output_size))

                # 初始化参数
                self._init_parameters()

            def _init_parameters(self):
                """初始化网络参数"""
                # 使用He初始化
                nn.init.kaiming_normal_(self.weights)
                nn.init.kaiming_normal_(self.output_weights)
                
                # 初始化偏置为小随机值
                nn.init.uniform_(self.biases, -0.1, 0.1)
                nn.init.uniform_(self.output_bias, -0.1, 0.1)
                nn.init.uniform_(self.p_values, -0.1, 0.1)

            def _compute_smooth_transition(self, x):
                """计算平滑过渡函数"""
                left_transition = torch.sigmoid(self.k * (x - self.range_min))
                right_transition = torch.sigmoid(self.k * (self.range_max - x))
                return left_transition * right_transition

            def forward(self, x):
                # 分离特殊变量
                special_x = x[:, self.special_index]
                
                # 计算过渡函数
                smooth_mask = self._compute_smooth_transition(special_x)
                
                # 为每个神经元计算隐藏层输出
                hidden_outputs = []
                for i in range(self.hidden_size):
                    # 复制输入以保持维度一致
                    x_modified = x.clone()
                    
                    # 对特殊变量应用平滑过渡
                    modified_special_x = smooth_mask * self.p_values[i] + (1 - smooth_mask) * special_x
                    x_modified[:, self.special_index] = modified_special_x
                    
                    # 计算该神经元的输出
                    neuron_output = torch.sum(self.weights[i] * x_modified, dim=1) + self.biases[i]
                    hidden_outputs.append(neuron_output)
                
                # 将所有神经元的输出堆叠
                hidden_layer = torch.stack(hidden_outputs, dim=1)
                
                # 应用ReLU激活函数
                hidden_layer = torch.relu(hidden_layer)
                
                # 更新self.hidden的权重和偏置
                self.hidden.weight.data = self.weights
                self.hidden.bias.data = self.biases
                
                # 更新self.output的权重和偏置
                self.output.weight.data = self.output_weights
                self.output.bias.data = self.output_bias
                
                # 计算输出层
                output = torch.mm(hidden_layer, self.output_weights.t()) + self.output_bias
                
                return output


        # 参数检查
        if not 0 <= special_index < input_size:
            raise ValueError(f"special_index必须在0到{input_size-1}之间")
        if range_max <= range_min:
            raise ValueError("range_max必须大于range_min")
        
        # 创建网络
        network = CustomNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            special_index=special_index,
            range_min=range_min,
            range_max=range_max
        )
        
        return network

    def train_control_model(self, n, trials=10, epochs=5000,special_index=0, range_min=0, range_max=1, learning_rate=0.1, min_lr=0.01, switch_threshold=0.01, patience=1000, min_delta=1e-7, decay_rate=[0.8,50], pop_ratio=0.5):
        """
        训练神经网络trials次，返回每次训练的MSE、预测值、折点位置和残差。
        
        参数:
            n: int, 神经元数量
            trials: int, 训练次数
            其他参数与neuron_mse_analysis相同
            pop_ratio: float, 要删除的trials的比例
            
        返回:

        """
        import copy
        
        mse_list = []  # 存储每次训练的MSE
        best_models_list = []  # 存储每次训练的最佳模型
        #param_cov_list = []  # 存储每次训练的参数方差-协方差矩阵
        
        # 设置批量大小
        batch_size = min(512, len(self.X))
        total_batches = len(self.X) // batch_size + (1 if len(self.X) % batch_size != 0 else 0)
        
        trial = 0
        while trial < trials:
            print(f"\n第 {trial+1}/{trials} 次训练")
            print(f"神经元数量: {n}")
            print(f"学习率: {learning_rate}")
            
            # 创建模型
            input_size = len(self.X_cols)
            self.model = self.create_control_network(input_size=input_size, hidden_size=n, output_size=1, special_index=special_index, range_min=range_min, range_max=range_max)
            
            # 准备数据批次
            perm = torch.randperm(len(self.X_tensor))
            batch_indices = [(i*batch_size, min((i+1)*batch_size, len(self.X))) for i in range(total_batches)]
            X_batches = [self.X_tensor[perm[start:end]] for start, end in batch_indices]
            y_batches = [self.y_tensor[perm[start:end]] for start, end in batch_indices]
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=decay_rate[0], 
                patience=decay_rate[1], verbose=False, min_lr=min_lr
            )
            
            # 训练过程变量初始化
            recent_losses = []
            stage = 1
            best_model = None
            best_loss = float('inf')
            epochs_since_best = 0
            need_retrain = False
            
            # 用于收集梯度的列表
            #gradients = []
            
            # 训练模型
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for b in range(total_batches):
                    optimizer.zero_grad()
                    outputs = self.model(X_batches[b])
                    loss = criterion(outputs, y_batches[b])
                    loss.backward()
                    
                    # 收集梯度
                    #current_gradients = []
                    #for param in self.model.parameters():
                    #    if param.grad is not None:
                    #        current_gradients.append(param.grad.view(-1))
                    #gradients.append(torch.cat(current_gradients))
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_loss /= total_batches
                recent_losses.append(epoch_loss)

                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f},  epochs since best: {epochs_since_best}")

                # 第100轮loss检查
                if epoch == 99 and epoch_loss > 10:
                    need_retrain = True
                    print(f"第100轮loss为{epoch_loss:.4f}，大于10，需要重新训练")
                    break
                
                # 更新最佳模型
                if epoch_loss < (best_loss - min_delta):
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(self.model)
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                
                # 学习率调度
                if stage == 1:
                    scheduler1.step(epoch_loss)
                    if len(recent_losses) >= 107:
                        recent_change = abs(np.mean(recent_losses[-6:-1]) - np.mean(recent_losses[-106:-101])) / np.mean(recent_losses[-106:-101])
                        if recent_change < switch_threshold:
                            stage = 2
                            print(f"切换到第二阶段，当前epoch: {epoch+1}")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=epoch, eta_min=min_lr
                            )
                else:
                    scheduler2.step()
                
                # 早停
                if epochs_since_best >= patience:
                    print(f"早停触发，{patience}轮未改善")
                    break
            
            if need_retrain:
                continue
                
           # 使用最佳模型进行评估
            self.model = best_model

            # 计算参数的方差-协方差矩阵
            #gradients = torch.stack(gradients).cpu()  # 将梯度移至CPU
            #mean_gradients = gradients.mean(dim=0, keepdim=True)  # 计算梯度均值
            #centered_gradients = gradients - mean_gradients  # 中心化梯度
            #param_cov = torch.mm(centered_gradients.t(), centered_gradients) / (len(gradients) - 1)  # 无偏协方差矩阵

            # 将最佳模型和方差-协方差矩阵添加到列表中
            #param_cov_list.append(param_cov)

            # 将最佳模型添加到列表中
            best_models_list.append(copy.deepcopy(best_model))
            with torch.no_grad():
                y_pred = self.model(self.X_tensor)
                mse = criterion(y_pred, self.y_tensor).item()
                print(f"最终MSE: {mse:.6f}")
                mse_list.append(mse)
                

            
            trial += 1
            
        # 根据MSE筛选最好的trials
        sorted_indices = np.argsort(mse_list)  # 按MSE排序
        keep_trials = max(1,int(trials*pop_ratio))  
        kept_indices = sorted_indices[:keep_trials]  # 保留的trial索引
        print(f"\n保留表现最好的 {keep_trials} 次训练结果")
        


        best_models = [best_models_list[i] for i in kept_indices]
        mse_list = [mse_list[i] for i in kept_indices]
            


        
        return  best_models,mse_list

    def analyze_moderation_effect(self, results, x_name):
        """
        分析调节效应的函数。
        
        参数:
            results: 包含模型训练结果的元组，包括MSE列表和模型列表
            x_name: 字符串，要分析的自变量名称
            
        功能:
            1. 分析指定自变量对其他变量的调节效应
            2. 通过在不同区间计算条件均值来评估调节作用
            3. 对结果进行标准化和平滑处理
            4. 可视化调节效应的变化趋势
            
        返回:
            无直接返回值，但会生成相关的分析结果和可视化图表
            该sum_std_list的值越大，说明调节效应越强，同时针对不同的变量进行调节效应分析的时候，对输出结果只能进行内部比较，而不能进行外部比较
        """
        mse_list = results[0]
        model_list = results[1]
        # 创建一个新的tensor，复制原始数据
        modified_x = self.X_tensor.clone()
        # 获取指定列的范围并计算区间
        x_index = self.X_cols.index(x_name)
        x_min = modified_x[:, x_index].min()
        x_max = modified_x[:, x_index].max()
        intervals = np.linspace(x_min, x_max, 11)  # 10个区间需要11个边界点
        bin_centers = [(intervals[i] + intervals[i+1])/2 for i in range(len(intervals)-1)]
        
        # 创建一个与X_cols等长的索引列表,排除x_index位置的索引
        control_x_indices = [i for i in range(len(self.X_cols)) if i != x_index]
        control_x_name = [self.X_cols[i] for i in control_x_indices]
        sum_std_list = []


        for l in control_x_indices:
            # 存储每个bin center的条件均值
            slope_list = []
            portion_list = []
            interval_points_num_list = []

            # 对每个bin center进行循环
            for i in range(len(bin_centers)-1):
                # 将指定列设置为当前bin center的值
                # 筛选出在当前区间内的点的索引

                mask = (self.X_tensor[:, x_index] >= intervals[i]) & (self.X_tensor[:, x_index] < intervals[i+1])
                interval_points = modified_x[mask]
                interval_points[:, x_index] = bin_centers[i]
                
                # 计算该bin center下的条件均值
                conditional_MEAN,conditional_MEAN_PORTION,interval_points_num = self.calculate_conditional_means(mse_list,model_list, interval_points, x_index, l)
                slope_list.append(conditional_MEAN)
                portion_list.append(conditional_MEAN_PORTION)
                interval_points_num_list.append(interval_points_num)

            # 将slope_list的行列互换
            slope_list_T = list(map(list, zip(*slope_list)))
            from scipy.interpolate import UnivariateSpline
            smooth_slope_list_T = []
            for K in slope_list_T:
                # 筛选出非空值及其对应的索引位置
                valid_idx = [i for i,v in enumerate(K) if not np.isnan(v)]
                valid_K = [K[i] for i in valid_idx]
                # 使用非空值进行样条拟合
                spline = UnivariateSpline(valid_idx, valid_K, s=9)  # s参数控制平滑程度
                # 预测完整序列的平滑值
                K_smooth = spline(list(range(len(K))))
                smooth_slope_list_T.append(K_smooth)
            smooth_slope_list = list(map(list, zip(*smooth_slope_list_T)))


            # 将二维的slope_list展平成一维数组
            flattened_slopes = [slope for sublist in smooth_slope_list for slope in sublist]
            
            # 计算以0为中心的标准差
            std_dev = np.sqrt(np.mean(np.array(flattened_slopes)**2))  # 计算标准差
            smooth_slope_list = [[slope / std_dev for slope in slopes] for slopes in smooth_slope_list]


            STDS = []
            for slope_list,portion_list in zip(smooth_slope_list,portion_list):
                # 使用conditional_portion对conditional_means加权求和
                weighted_mean = sum(m * p for m, p in zip(slope_list, portion_list) if not np.isnan(m))

                # 计算加权标准差
                weighted_std = np.sqrt(sum(p * (m - weighted_mean)**2 for m, p in zip(slope_list, portion_list) if not np.isnan(m)))
                STDS.append(weighted_std)

            interval_points_num_list = [i/np.sum(interval_points_num_list) for i in interval_points_num_list]
            sum_std_list.append(np.sum(np.array(STDS) * np.array(interval_points_num_list)))

            '''
            std_list = []
            for i in smooth_slope_list:
                weight = interval_points_num/np.sum(interval_points_num)
                weighted_slope = np.average(i,weights=weight)
                # 计算加权标准差
                weighted_std = np.sqrt(np.average((i - weighted_slope)**2, weights=weight))
                std_list.append(weighted_std)

            sum_std_list.append(np.sum(std_list))'''
        # 将sum_std_list和control_x_name打包在一起并按sum_std_list降序排序
        sorted_pairs = sorted(zip(sum_std_list, control_x_name), reverse=True)
        # 解压缩排序后的配对
        sum_std_list, control_x_name = zip(*sorted_pairs)
        # 转换回列表
        sum_std_list = list(sum_std_list)
        # 标准化sum_std_list到0-1区间

        max_val = max(sum_std_list)
        sum_std_list = [round(x / max_val, 3) for x in sum_std_list]
        control_x_name = list(control_x_name)

        
        # 创建DataFrame
        result_df = pd.DataFrame({
            '变量名': control_x_name,
            '调节效应指数': sum_std_list
        })
        
        # 打印表格
        print("\n调节效应分析结果:")
        from tabulate import tabulate
        print(tabulate(result_df, headers='keys', tablefmt='pretty', showindex=False))





        return result_df



    def calculate_conditional_means(self,mse_list, model_list, modified_x,main_x_idx,control_x_idx):
        """
        计算在控制变量不同区间下的条件均值
        
        参数:
            model: 神经网络模型
            modified_x: numpy.ndarray, 输入数据矩阵
            control_x_idx: int, 控制变量的索引位置
            
        返回:
            bin_centers: numpy.ndarray, 每个区间的中点值
            conditional_means: numpy.ndarray, 每个区间的条件均值
        """
        MEAN_PREDICTIONS = []
        MEAN_PORTION = []
        STDS = []
        # 获取控制变量的值
        control_var = modified_x[:, control_x_idx]
        lower_bound = np.percentile(control_var, 2.5)
        upper_bound = np.percentile(control_var, 97.5)


        # 计算区间边界
        bins = np.linspace(lower_bound, upper_bound, 11)  # 10个区间需要11个边界点
        bin_centers = [bins[i] + bins[i+1] / 2 for i in range(len(bins)-1)]


        interval_points_total_num = len(control_var[(control_var >= lower_bound) & (control_var <= upper_bound)])


        for model in model_list:
            # 存储每个区间的条件均值
            conditional_means = []
            conditional_portion = []
            # 对每个区间进行处理
            for i in range(len(bins)-1):
                # 获取当前区间的掩码
                mask = (control_var >= bins[i]) & (control_var < bins[i+1])
                if i == len(bins)-2:  # 处理最后一个区间，包含右端点
                    mask = (control_var >= bins[i]) & (control_var <= bins[i+1])
                    
                # 获取当前区间的数据点
                interval_points = modified_x[mask]
                interval_points_proportion = len(interval_points)
                if len(interval_points) > 0:  # 如果区间内有点
                    # 将该区间内所有点的control_x_idx位置设为区间中点
                    interval_points[:, control_x_idx] = torch.tensor(bin_centers[i])
                    interval_points_ = interval_points.clone()
                    interval_points_[:, main_x_idx] = interval_points_[:, main_x_idx] + 0.0001

                    
                    # 转换为tensor并进行预测
                    with torch.no_grad():
                        predictions = model(torch.FloatTensor(interval_points))
                        predictions_ = model(torch.FloatTensor(interval_points_))
                    PREDICTIONS = (predictions_[0] - predictions[0])/0.0001
                    # 计算该区间的预测均值
                    mean_prediction = PREDICTIONS.mean().item()
                else:
                    mean_prediction = np.nan
                    interval_points_proportion = 0
                    
                conditional_means.append(mean_prediction)
                conditional_portion.append(interval_points_proportion)
            # 对conditional_portion进行归一化
            total_points = sum(conditional_portion)
            conditional_portion = [p/total_points for p in conditional_portion]

            # 使用conditional_portion对conditional_means加权求和
            weighted_mean = sum(m * p for m, p in zip(conditional_means, conditional_portion) if not np.isnan(m))

            # 计算加权标准差
            weighted_std = np.sqrt(sum(p * (m - weighted_mean)**2 for m, p in zip(conditional_means, conditional_portion) if not np.isnan(m)))
            STDS.append(weighted_std)

            MEAN_PREDICTIONS.append(conditional_means)
            MEAN_PORTION.append(conditional_portion)
        # 将多个模型的预测结果取平均
        weights = np.array([1/mse for mse in mse_list])
        weights = weights / np.sum(weights)  # 归一化权重
        conditional_STD = np.average(STDS, axis=0, weights=weights)
        conditional_MEAN = np.average(MEAN_PREDICTIONS, axis=0, weights=weights)
        conditional_MEAN_PORTION = np.average(MEAN_PORTION, axis=0, weights=weights)

        
        return conditional_MEAN,conditional_MEAN_PORTION,interval_points_total_num




    def plot_moderation_effect(self, results, main_x_name, control_x_name,caculate_moderation_LRT=False,train_control_model_setting = [10,5000,0.1, 0.01, 0.01, 1000, 1e-7, [0.8,50],0.5]):
        """
        绘制调节效应图并计算调节效应的似然比检验
        当两个变量存在较明显的共线性问题时，就几乎不会存在调节效应
        参数:
            results (tuple): 包含训练结果的元组,包括:
                - mse_list: 每个模型的MSE列表
                - model_list: 训练好的模型列表
                - n: 神经元数量
            main_x_name (str): 主要自变量的列名
            control_x_name (str): 调节变量的列名
            caculate_moderation_LRT (bool): 是否计算调节效应的似然比检验,默认为False
            train_control_model_setting (list): 训练控制模型的参数设置,包含:
                - trials: 训练次数
                - epochs: 每次训练的迭代次数
                - learning_rate: 初始学习率
                - min_lr: 最小学习率
                - switch_threshold: 切换阈值
                - patience: 早停耐心值
                - min_delta: 最小变化量
                - decay_rate: 学习率衰减参数[衰减因子,耐心值]
                
        返回:
            result_dict (dict): 包含条件均值预测结果的字典
                - key: 调节变量的取值点
                - value: 包含该点处的条件均值和主变量取值点的字典
            best_model: 表现最好的模型
        """
        
        mse_list = results[0]
        model_list = results[1]
        n = results[2]
        # 创建一个新的tensor，复制原始数据
        x = self.X_tensor.clone()
        # 获取指定列的范围并计算区间
        main_x_index = self.X_cols.index(main_x_name)
        # 获取main_x的标准化范围
        main_x = x[:, main_x_index]
        main_x_min = main_x.min().item()
        main_x_max = main_x.max().item()
        
        # 还原到原始空间
        main_x_std = self.X_std[main_x_index] 
        main_x_mean = self.X_mean[main_x_index]
        main_x_min_orig = main_x_min * main_x_std + main_x_mean
        main_x_max_orig = main_x_max * main_x_std + main_x_mean
        
        # 在原始空间范围内生成100个均匀分布的点
        main_x_points = np.linspace(main_x_min_orig, main_x_max_orig, 100)
        control_x_index = self.X_cols.index(control_x_name)



        conditional_mp,ALPHAS,bin_centers = self.calculate_conditional_predictions(mse_list, model_list, x,main_x_index,control_x_index)
        # 还原conditional_mp到原始空间
        conditional_mp = [mp * self.y_std + self.y_mean for mp in conditional_mp]
        
        # 还原bin_centers到原始空间 
        control_x_std = self.X_std[control_x_index]
        control_x_mean = self.X_mean[control_x_index]
        bin_centers = np.array(bin_centers) * control_x_std + control_x_mean
        
        # 创建结果字典
        result_dict = {
            bin_centers[i]: {
                'conditional_means': conditional_mp[i],
                'main_x_points': main_x_points,
                'alphas': ALPHAS[i]
            } for i in range(5)
        }

        y_true = self.y_tensor.numpy()
        x_tensor = torch.tensor(x, dtype=torch.float32)


        weights = np.array([1/mse**2 for mse in mse_list])
        weights = weights / np.sum(weights)  # 归一化权重

        # 使用weights对mse_list进行加权求和
        loss = np.sum([mse * weight for mse, weight in zip(mse_list, weights)])
        '''with torch.no_grad():
            main_y_preds = [model(x_tensor)[0].numpy() for model in model_list]
            residuals_main = [y_true - pred for pred in main_y_preds]
            weighted_residuals_main = np.sum([res * weight for res, weight in zip(residuals_main, weights)], axis=0)
        main_rss = np.sum((weighted_residuals_main) ** 2)'''



        # 绘制图表
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体为微软雅黑
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        if caculate_moderation_LRT:
            nn =int( n*(len(self.X_cols)-1)/len(self.X_cols))
            # 训练调节效应的控制模型
            control_models,mse_list_control = self.train_moderation_control_model( nn, main_x_index, control_x_index,trials=train_control_model_setting[0],
                                                                                   epochs=train_control_model_setting[1],
                                                                                   learning_rate=train_control_model_setting[2],
                                                                                    min_lr=train_control_model_setting[3],
                                                                                    switch_threshold=train_control_model_setting[4],
                                                                                   patience=train_control_model_setting[5], 
                                                                                   min_delta=train_control_model_setting[6], 
                                                                                   decay_rate=train_control_model_setting[7],
                                                                                   pop_ratio=train_control_model_setting[8])
            weights_control = np.array([1/mse**2 for mse in mse_list_control])
            weights_control = weights_control / np.sum(weights_control)  # 归一化权重
            loss_control = np.sum([mse * weight for mse, weight in zip(mse_list_control, weights_control)])
            
            # 计算控制模型的RSS

            '''with torch.no_grad():
                control_y_preds = [model(x_tensor)[0].numpy() for model in control_models]
                residuals_control = [y_true - pred for pred in control_y_preds]
                weighted_residuals_control = np.sum([res * weight for res, weight in zip(residuals_control, weights_control)], axis=0)

            control_rss = np.sum((weighted_residuals_control) ** 2)
            
            # 计算似然比统计量
            
            main_sigma2 = main_rss / N
            control_sigma2 = control_rss / N'''
            N = len(y_true)
            likelihood_ratio = N * np.log(loss_control/loss)
            
            if likelihood_ratio < 0:
                print("likelihood_ratio≈0，建议多次尝试或增加训练次数")
                likelihood_ratio = 0
                
            if likelihood_ratio > 10.828:
                plt.text(0.5, 0.80, f"LRT≈{likelihood_ratio:.3f}\nLRT>10.828\np<0.001", 
                        transform=plt.gca().transAxes, color='green', ha='center')
            elif likelihood_ratio > 6.635:
                plt.text(0.5, 0.80, f"LRT≈{likelihood_ratio:.3f}\nLRT>6.635\np<0.01",
                        transform=plt.gca().transAxes, color='yellowgreen', ha='center')
            elif likelihood_ratio > 3.841:
                plt.text(0.5, 0.80, f"LRT≈{likelihood_ratio:.3f}\nLRT>3.841\np<0.05",
                        transform=plt.gca().transAxes, color='yellow', ha='center')
            elif likelihood_ratio > 2.706:
                plt.text(0.5, 0.80, f"LRT≈{likelihood_ratio:.3f}\nLRT>2.706\np<0.1",
                        transform=plt.gca().transAxes, color='#04D462', ha='center')
            else:
                plt.text(0.5, 0.80, f"LRT≈{likelihood_ratio:.3f}\nLRT<2.706\np>0.1",
                        transform=plt.gca().transAxes, color='gray', ha='center')







        colors = ['#010FE5', '#00FAC7', '#DCFA41', '#FAAE40', '#F95133']

        for i in range(5):
            for j in range(len(main_x_points)-1):
                plt.plot(main_x_points[j:j+2], 
                        result_dict[bin_centers[i]]['conditional_means'][j:j+2],
                        color=colors[i], 
                        alpha=np.array(ALPHAS)[i][j]
                        )
        plt.xlabel(main_x_name)
        plt.ylabel(f'{self.y_col}的预测值')
        plt.title(f'调节效应：不同{control_x_name}下{main_x_name}对{self.y_col}的影响')
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], label=f'{control_x_name}={bin_centers[i]:.2f}') for i in range(5)]
        plt.legend(handles=legend_elements)


        plt.show()

        



        return result_dict



    def calculate_conditional_predictions(self,mse_list, model_list, x,main_x_idx,control_x_idx):
        """
        计算在控制变量不同区间下的条件均值
        控制变量区间为2.5%和97.5%分位数，排除了极端值
        
        参数:
            model: 神经网络模型
            modified_x: numpy.ndarray, 输入数据矩阵
            control_x_idx: int, 控制变量的索引位置
            
        返回:
            bin_centers: numpy.ndarray, 每个区间的中点值
            conditional_mp: numpy.ndarray, 每个区间的条件预测值
        """

        
        # 获取控制变量的值
        control_var = x[:, control_x_idx]
        # 获取control_var的2.5%和97.5%分位数
        lower_bound = np.percentile(control_var, 2.5)
        upper_bound = np.percentile(control_var, 97.5)
        main_var = x[:, main_x_idx]
        main_var_min = main_var.min()
        main_var_max = main_var.max()
        main_var_intervals = np.linspace(main_var_min, main_var_max, 100)
        dx = main_var_intervals[1] - main_var_intervals[0]



        # 计算区间边界
        bins = np.linspace(lower_bound, upper_bound, 6)  
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]




        from scipy.interpolate import UnivariateSpline
        CONDITIONAL_PREDICTIONS = []
        CONDITIONAL_ALPHAS = []
        for model in model_list:
            # 存储每个区间的条件均值
            conditional_mean_predictions = []
            conditional_alphas = []
            
            # 对每个控制变量的区间进行处理
            for i in range(len(bins)-1):
                # 获取当前区间的掩码
                mask = (control_var >= bins[i]) & (control_var < bins[i+1])
                if i == len(bins)-2:  # 处理最后一个区间，包含右端点
                    mask = (control_var >= bins[i]) & (control_var <= bins[i+1])
                points_in_interval = torch.sum(mask).item()
                    
                # 获取当前区间的数据点
                interval_points = x[mask]
                # 如果当前区间有数据点
                if torch.sum(mask) > 0:
                    predictions = []
                    alphas = []
                    # 对每个数据点进行预测
                    for i in range(len(main_var_intervals)):
                        # 计算当前区间的范围
                        interval_start = (i-2)*dx + main_var_min
                        interval_end = (i+2)*dx + main_var_min
                        
                        # 筛选出在当前区间范围内的点
                        interval_mask = (interval_points[:, main_x_idx] >= interval_start) & (interval_points[:, main_x_idx] < interval_end)
                        interval_data = interval_points[interval_mask]
                        interval_data_ = interval_data.clone()
                        interval_data_[:, main_x_idx] = (i)*dx + main_var_min
                        
                        alpha_start = (i-5)*dx + main_var_min
                        alpha_end = (i+5)*dx + main_var_min
                        alpha = torch.sum((interval_points[:, main_x_idx] >= alpha_start) & (interval_points[:, main_x_idx] < alpha_end))/points_in_interval
                        if alpha > 0.1:
                            alpha = 0.1
                        if alpha < 0.02:
                            alpha = 0
                        alpha = alpha*10
                        alphas.append(alpha)
                        
                        if len(interval_data) > 0:
                            # 使用模型进行预测
                            with torch.no_grad():
                                pred = model(interval_data_)
                            predictions.append(pred[0].mean().item())
                        else:
                            predictions.append(np.nan)
                    # 使用UnivariateSpline进行非线性插值
                    valid_indices = ~np.isnan(predictions)
                    if np.sum(valid_indices) > 3:  # 确保有足够的点进行插值
                        x_valid = np.arange(len(predictions))[valid_indices]
                        y_valid = np.array(predictions)[valid_indices]
                        # 使用LOWESS进行局部加权回归
                        from statsmodels.nonparametric.smoothers_lowess import lowess
                        # 对有效点进行LOWESS拟合
                        smoothed = lowess(y_valid, x_valid, frac=0.3, it=1, return_sorted=False)
                        # 使用线性插值填充NaN值
                        from scipy.interpolate import interp1d
                        f = interp1d(x_valid, smoothed, kind='linear', fill_value='extrapolate')
                        # 只对NaN值进行插值
                        nan_indices = np.isnan(predictions)
                        predictions = np.array(predictions)
                        predictions[nan_indices] = f(np.arange(len(predictions))[nan_indices])
                    conditional_mean_predictions.append(predictions)
                    conditional_alphas.append(alphas)
            CONDITIONAL_PREDICTIONS.append(conditional_mean_predictions)
            CONDITIONAL_ALPHAS.append(conditional_alphas)


                


                    

        # 将多个模型的预测结果取平均
        weights = np.array([1/mse for mse in mse_list])
        weights = weights / np.sum(weights)  # 归一化权重
        conditional_mp = []
        for l in range(len(conditional_mean_predictions)):
            a = []
            for i in range(len(CONDITIONAL_PREDICTIONS)):
                a.append(CONDITIONAL_PREDICTIONS[i][l])
            b = np.average(a, axis=0, weights=weights)
            from scipy.signal import savgol_filter
            b = savgol_filter(b, window_length=100, polyorder=3)
            conditional_mp.append(b)


        
        return conditional_mp,CONDITIONAL_ALPHAS[0],bin_centers




    def create_moderation_control_network(self, input_size, hidden_size, output_size, x1_index, x2_index):
        """
            创建调节效应控制网络的辅助函数
            
            参数:
                input_size (int): 输入特征的维度
                hidden_size (int): 每种神经元的数量
                output_size (int): 输出维度
                x1_index (int): 第一个指定变量的索引
                x2_index (int): 第二个指定变量的索引
                
            返回:
                ModerationControlNetwork实例
        """
        class ModerationControlNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, x1_index, x2_index):
                """
                初始化调节效应控制网络
                
                参数:
                    input_size (int): 输入特征的维度
                    hidden_size (int): 每种神经元的数量
                    output_size (int): 输出维度
                    x1_index (int): 第一个指定变量的索引
                    x2_index (int): 第二个指定变量的索引
                """
                super(ModerationControlNetwork, self).__init__()
                
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.x1_index = x1_index
                self.x2_index = x2_index
                
                # 创建两组神经元的权重和偏置
                # 第一组神经元不接收x1
                self.weights1 = nn.Parameter(torch.randn(hidden_size, input_size))
                self.bias1 = nn.Parameter(torch.randn(hidden_size))
                
                # 第二组神经元不接收x2
                self.weights2 = nn.Parameter(torch.randn(hidden_size, input_size))
                self.bias2 = nn.Parameter(torch.randn(hidden_size))
                
                # 输出层
                self.output_layer = nn.Linear(hidden_size * 2, output_size)
                
                # 激活函数
                self.relu = nn.ReLU()
                
                # 初始化参数
                self._init_parameters()
                
            def _init_parameters(self):
                """初始化网络参数"""
                # 使用He初始化
                nn.init.kaiming_normal_(self.weights1)
                nn.init.kaiming_normal_(self.weights2)
                nn.init.kaiming_normal_(self.output_layer.weight)
                
            def forward(self, x):
                """
                前向传播
                
                参数:
                    x: 输入张量 shape: (batch_size, input_size)
                    
                返回:
                    输出张量 shape: (batch_size, output_size)
                """
                batch_size = x.size(0)
                
                # 第一组神经元的计算（不使用x1）
                # 将x1对应的权重设为0
                weights1_masked = self.weights1.clone()
                weights1_masked[:, self.x1_index] = 0
                hidden1 = self.relu(torch.mm(x, weights1_masked.t()) + self.bias1)
                
                # 第二组神经元的计算（不使用x2）
                # 将x2对应的权重设为0
                weights2_masked = self.weights2.clone()
                weights2_masked[:, self.x2_index] = 0
                hidden2 = self.relu(torch.mm(x, weights2_masked.t()) + self.bias2)
                
                # 合并两组神经元的输出
                combined = torch.cat((hidden1, hidden2), dim=1)
                
                # 输出层
                output = self.output_layer(combined)
                
                return output


        # 验证输入参数
        if not 0 <= x1_index < input_size:
            raise ValueError(f"x1_index必须在0到{input_size-1}之间")
        if not 0 <= x2_index < input_size:
            raise ValueError(f"x2_index必须在0到{input_size-1}之间")
        if x1_index == x2_index:
            raise ValueError("x1_index和x2_index不能相同")
            
        # 创建网络
        network = ModerationControlNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                x1_index=x1_index,
                x2_index=x2_index
        )
            
        return network
            

    def train_moderation_control_model(self, n, x1_index, x2_index, trials=10, epochs=5000,learning_rate=0.1, min_lr=0.01, switch_threshold=0.01, patience=1000, min_delta=1e-7, decay_rate=[0.8,50],pop_ratio=0.5):
        """
        训练神经网络trials次，返回每次训练的MSE、预测值、折点位置和残差。
        
        参数:
            n: int, 神经元数量
            trials: int, 训练次数
            其他参数与neuron_mse_analysis相同
            
        返回:

        """
        import copy
        
        mse_list = []  # 存储每次训练的MSE
        best_models_list = []  # 存储每次训练的最佳模型
        #param_cov_list = []  # 存储每次训练的参数方差-协方差矩阵
        
        # 设置批量大小
        batch_size = min(512, len(self.X))
        total_batches = len(self.X) // batch_size + (1 if len(self.X) % batch_size != 0 else 0)
        
        trial = 0
        while trial < trials:
            print(f"\n第 {trial+1}/{trials} 次训练")
            print(f"神经元数量: {n}")
            print(f"学习率: {learning_rate}")
            
            # 创建模型
            input_size = len(self.X_cols)
            self.model = self.create_moderation_control_network(input_size=input_size, hidden_size=n, output_size=1, x1_index=x1_index, x2_index=x2_index)
            
            # 准备数据批次
            perm = torch.randperm(len(self.X_tensor))
            batch_indices = [(i*batch_size, min((i+1)*batch_size, len(self.X))) for i in range(total_batches)]
            X_batches = [self.X_tensor[perm[start:end]] for start, end in batch_indices]
            y_batches = [self.y_tensor[perm[start:end]] for start, end in batch_indices]
            
            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=decay_rate[0], 
                patience=decay_rate[1], verbose=False, min_lr=min_lr
            )
            
            # 训练过程变量初始化
            recent_losses = []
            stage = 1
            best_model = None
            best_loss = float('inf')
            epochs_since_best = 0
            need_retrain = False
            
            # 用于收集梯度的列表
            #gradients = []
            
            # 训练模型
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for b in range(total_batches):
                    optimizer.zero_grad()
                    outputs = self.model(X_batches[b])
                    loss = criterion(outputs, y_batches[b])
                    loss.backward()
                    
                    # 收集梯度
                    #current_gradients = []
                    #for param in self.model.parameters():
                    #    if param.grad is not None:
                    #        current_gradients.append(param.grad.view(-1))
                    #gradients.append(torch.cat(current_gradients))
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_loss /= total_batches
                recent_losses.append(epoch_loss)

                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f},  epochs since best: {epochs_since_best}")

                # 第100轮loss检查
                if epoch == 99 and epoch_loss > 10:
                    need_retrain = True
                    print(f"第100轮loss为{epoch_loss:.4f}，大于10，需要重新训练")
                    break
                
                # 更新最佳模型
                if epoch_loss < (best_loss - min_delta):
                    best_loss = epoch_loss
                    best_model = copy.deepcopy(self.model)
                    epochs_since_best = 0
                else:
                    epochs_since_best += 1
                
                # 学习率调度
                if stage == 1:
                    scheduler1.step(epoch_loss)
                    if len(recent_losses) >= 107:
                        recent_change = abs(np.mean(recent_losses[-6:-1]) - np.mean(recent_losses[-106:-101])) / np.mean(recent_losses[-106:-101])
                        if recent_change < switch_threshold:
                            stage = 2
                            print(f"切换到第二阶段，当前epoch: {epoch+1}")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = learning_rate
                            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=epoch, eta_min=min_lr
                            )
                else:
                    scheduler2.step()
                
                # 早停
                if epochs_since_best >= patience:
                    print(f"早停触发，{patience}轮未改善")
                    break
            
            if need_retrain:
                continue
                
           # 使用最佳模型进行评估
            self.model = best_model

            # 计算参数的方差-协方差矩阵
            #gradients = torch.stack(gradients).cpu()  # 将梯度移至CPU
            #mean_gradients = gradients.mean(dim=0, keepdim=True)  # 计算梯度均值
            #centered_gradients = gradients - mean_gradients  # 中心化梯度
            #param_cov = torch.mm(centered_gradients.t(), centered_gradients) / (len(gradients) - 1)  # 无偏协方差矩阵

            # 将最佳模型和方差-协方差矩阵添加到列表中
            #param_cov_list.append(param_cov)

            # 将最佳模型添加到列表中
            best_models_list.append(copy.deepcopy(best_model))
            with torch.no_grad():
                y_pred = self.model(self.X_tensor)
                mse = criterion(y_pred, self.y_tensor).item()
                print(f"最终MSE: {mse:.6f}")
                mse_list.append(mse)
                

            
            trial += 1
            
        # 根据MSE筛选最好的trials
        sorted_indices = np.argsort(mse_list)  # 按MSE排序
        keep_trials =max(int(trials*pop_ratio),1)  
        kept_indices = sorted_indices[:keep_trials]  # 保留的trial索引
        print(f"\n保留表现最好的 {keep_trials} 次训练结果")
        


        best_models = [best_models_list[i] for i in kept_indices] 
        mse_list = [mse_list[i] for i in kept_indices]


        
        return  best_models,mse_list
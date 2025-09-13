# coding: UTF-8
# coding: UTF-8
import time

print("1. time imported")
import torch

print("2. torch imported")
import numpy as np

print("3. numpy imported")
from train_eval import train, init_network

print("4. train_eval imported")
from importlib import import_module

print("5. import_module imported")
import argparse

print("6. argparse imported")
from utils import build_dataset, build_iterator, get_time_dif

print("7. utils imported")

# --- 修改点 1: 在这里添加新的命令行参数 ---
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: bert_gru_attention, Bert, ERNIE, etc.')

# 新增 alpha 参数，用于控制损失权重
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight for the hard-label loss (CrossEntropy). Range [0, 1]. Default: 0.3')

# 新增 temperature 参数，用于软化概率
parser.add_argument('--temperature', type=float, default=2.0,
                    help='Temperature for softening probabilities in distillation. Default: 5.0')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    # --- 修改点 2: 将命令行传入的参数赋值给 config 对象 ---
    # 这样，train_eval.py 里面的 train 函数就能通过 config.distillation_alpha 访问到它
    config.distillation_alpha = args.alpha
    config.distillation_temperature = args.temperature
    # ----------------------------------------------------

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print(">>> 脚本开始，准备加载数据...")  # <--- 添加这行
    train_data, dev_data, test_data = build_dataset(config)
    print(">>> 数据加载完毕，准备构建迭代器...")  # <--- 添加这行

    # 修改1：train需要teacher——probs，dev和test不需要
    train_iter = build_iterator(train_data, config, shuffle=True, need_teacher_probs=True)
    dev_iter = build_iterator(dev_data, config, shuffle=False)
    test_iter = build_iterator(test_data, config, shuffle=False)
    print(">>> 迭代器构建完毕，准备初始化模型...")  # <--- 添加这行

    time_dif = get_time_dif(start_time)
    print("Time usage for data loading:", time_dif)

    # --- 修改点 3: (推荐) 打印本次实验的配置信息 ---
    print("\n--- Experiment Configuration ---")
    print(f"  Model: {model_name}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Distillation Alpha (硬标签权重): {config.distillation_alpha}")
    print(f"  Distillation Temperature (温度): {config.distillation_temperature}")
    print("--------------------------------\n")
    # -----------------------------------------------

    # train
    model = x.Model(config).to(config.device)
    print(">>> 模型初始化完毕，即将进入训练函数...")  # <--- 添加这行    # init_network(model) # 如果你的模型需要特定的初始化，可以取消这行注释
    train(config, model, train_iter, dev_iter, test_iter)
# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, evaluate, test  # 导入 train, evaluate, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import gc  # 导入 gc 模块

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: bert_gru_attention, Bert, ERNIE, etc.')
parser.add_argument('--temperature', type=float, default=2.0,
                    help='Temperature for softening probabilities in distillation. Default: 5.0')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config.distillation_temperature = args.temperature

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print(">>> 脚本开始，准备加载数据...")
    train_data, dev_data, test_data = build_dataset(config)
    print(">>> 数据加载完毕，准备构建迭代器...")

    train_iter = build_iterator(train_data, config, shuffle=True, need_teacher_probs=True)
    dev_iter = build_iterator(dev_data, config, shuffle=False)
    test_iter = build_iterator(test_data, config, shuffle=False)
    print(">>> 迭代器构建完毕，准备初始化模型...")

    time_dif = get_time_dif(start_time)
    print("Time usage for data loading:", time_dif)

    # --- 修改点 1: 循环遍历不同的 alpha 值 ---
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    for alpha in alpha_values:
        print(f"\n--- Training with Distillation Alpha = {alpha} ---")
        config.distillation_alpha = alpha

        print("\n--- Experiment Configuration ---")
        print(f"  Model: {model_name}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Distillation Alpha (硬标签权重): {config.distillation_alpha}")
        print(f"  Distillation Temperature (温度): {config.distillation_temperature}")
        print("--------------------------------\n")

        # train
        model = x.Model(config).to(config.device)
        print(">>> 模型初始化完毕，即将进入训练函数...")
        model_path = f'{config.save_path}_alpha_{alpha:.1f}.ckpt'
        train(config, model, train_iter, dev_iter, test_iter, model_path=model_path)

        # ---  在测试集上评估模型并保存结果 ---
        print("\n--- Evaluating on Test Set ---")
        model = x.Model(config).to(config.device)  # 重新加载模型
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
        print(f"Test Results with Alpha = {alpha}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print("  Precision, Recall, and F1-Score:")
        print(test_report)

        results[alpha] = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_report': test_report,
            'test_confusion': test_confusion
        }

        # --- 删除模型并释放内存 ---
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # --- 修改点 3: 打印所有 alpha 值的实验结果 ---
    print("\n--- Summary of Results ---")
    for alpha, result in results.items():
        print(f"\nResults with Alpha = {alpha}:")
        print(f"  Test Loss: {result['test_loss']:.4f}")
        print(f"  Test Acc: {result['test_acc']:.4f}")
        print("  Precision, Recall, and F1-Score:")
        print(result['test_report'])
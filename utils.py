import torch
from tqdm import tqdm
import time
from datetime import timedelta
import random

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(dataset_path, pad_size=32, need_teacher_probs=False):
        contents = []
        with open(dataset_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f, desc=f"Loading data from {dataset_path}"):  # 添加一个描述，方便调试
                lin = line.strip()
                if not lin:
                    continue

                parts = lin.split('\t')
                if need_teacher_probs:
                    if len(parts) != 3:
                        print(f"Warning: Skipping malformed line in {dataset_path}: {lin}")
                        continue
                    content, label, teacher_probs_str = parts
                else:
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line in {dataset_path}: {lin}")
                        continue
                    content, label = parts
                    teacher_probs_str = None

                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                try:
                    if teacher_probs_str:
                        probs_list = [float(p) for p in teacher_probs_str.strip('[]').split(',')]
                        if len(probs_list) != config.num_classes:  # 检查类别数是否匹配
                            print(
                                f"Warning: Probability dimension mismatch in {dataset_path}. Expected {config.num_classes}, got {len(probs_list)}. Line: {lin}")
                            continue
                    else:
                        probs_list = None
                except ValueError:
                    print(f"Warning: Could not parse probabilities in {dataset_path}. Line: {lin}")
                    continue

                contents.append((token_ids, int(label), seq_len, mask, probs_list))
        return contents

    train = load_dataset(config.train_path, config.pad_size, need_teacher_probs=True)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, shuffle=True, need_teacher_probs=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        self.need_teacher_probs = need_teacher_probs
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.shuffle = shuffle

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        if self.need_teacher_probs:
            teacher_probs = torch.FloatTensor([_[4] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y, teacher_probs
        else:
            return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        self.index = 0
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, shuffle=True, need_teacher_probs=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, shuffle, need_teacher_probs)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
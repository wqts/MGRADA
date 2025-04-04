import os
import collections
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from lib.load_data import load_data
from lib.model import Model
from lib.model import Model, DomainDiscriminator, DomainAdversarialLoss
from lib.pipeline import main
from lib.utils import test, get_batch_indices, sharpen

WorkerDoneData = collections.namedtuple("WorkerDoneData", ("best_accuracy"))

def update_teacher_model(student_model, teacher_model, alpha):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def label_propagation(teacher_feature_lb, teacher_feature_ulb, teacher_feature_tar, y_lb, n_class, sigma=0.25, gamma=0.99):
    # 计算相似度矩阵
    features = torch.cat((teacher_feature_lb, teacher_feature_ulb, teacher_feature_tar), dim=0)
    n, d = features.shape
    device = features.device

    emb_all = features / (sigma + np.finfo(float).eps)
    emb1 = torch.unsqueeze(emb_all, 1)
    emb2 = torch.unsqueeze(emb_all, 0)
    w = ((emb1 - emb2) ** 2).mean(2)
    w = torch.exp(-w / 2)

    topk = 5
    topk_values, indices = torch.topk(w, topk)
    mask = torch.zeros_like(w).to(device)
    mask = mask.scatter(1, indices, 1)
    mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
    w = w * mask

    d = w.sum(0)
    d_sqrt_inv = torch.sqrt(1.0 / (d + np.finfo(float).eps)).to(device)
    d1 = torch.unsqueeze(d_sqrt_inv, 1).repeat(1, n)
    d2 = torch.unsqueeze(d_sqrt_inv, 0).repeat(n, 1)
    s = d1 * w * d2

    y = torch.zeros(len(y_lb) + len(teacher_feature_ulb) + len(teacher_feature_tar), n_class).to(device)
    y[:len(y_lb), :] = F.one_hot(y_lb, num_classes=n_class).float()
    
    # 迭代方法
    f = y.clone()
    for _ in range(10):  # 迭代次数
        f = gamma * torch.matmul(s, f) + (1 - gamma) * y

    pseudo_labels = torch.softmax(f, dim=1)

    pseudo_labels_ulb = pseudo_labels[len(y_lb):len(y_lb) + len(teacher_feature_ulb)]
    pseudo_labels_tar = pseudo_labels[len(y_lb) + len(teacher_feature_ulb):]

    return pseudo_labels_ulb, pseudo_labels_tar

def get_curriculum_pseudo_labels(config, student_logits, teacher_pseudo_labels, n_class, threshold_min=0.5, threshold_max=0.9):
    assert len(student_logits) == len(teacher_pseudo_labels)

    # 计算 eql_list 和 tea_list
    stu_preds = student_logits.argmax(dim=1)
    tea_preds = teacher_pseudo_labels.argmax(dim=1)
    eql_mask = (stu_preds == tea_preds)
    eql_list = torch.bincount(tea_preds[eql_mask], minlength=n_class).float().to(config["device"])
    tea_list = torch.bincount(tea_preds, minlength=n_class).float().to(config["device"]) + 1

    # 计算阈值
    threshold = threshold_min + (threshold_max - threshold_min) * (eql_list / tea_list)

    # 应用阈值过滤
    max_probs, tea_preds = teacher_pseudo_labels.max(dim=1)
    mask = max_probs >= threshold[tea_preds]
    teacher_pseudo_labels = teacher_pseudo_labels * mask.unsqueeze(1).float()

    return teacher_pseudo_labels

def train_epoch(config, dset_dict, student_model, teacher_model, global_domain_adv, sub_domain_advs, optimizer):
    student_model.train()
    teacher_model.train()
    global_domain_adv.train()

    # 初始化队列
    queue_size = config["queue_size"]
    feature_lb_queue = collections.deque(maxlen=queue_size)
    feature_ulb_queue = collections.deque(maxlen=queue_size)
    feature_tar_queue = collections.deque(maxlen=queue_size)
    y_lb_queue = collections.deque(maxlen=queue_size)

    for i in range(config["it_per_epoch"]):
        lb_indices = get_batch_indices(config, dset_dict['lb_dset'], config["batch_size"])
        ulb_indices = get_batch_indices(config, dset_dict['ulb_dset'], config["batch_size"])
        tar_indices = get_batch_indices(config, dset_dict['tar_dset'], config["batch_size"])
        x_lb = torch.index_select(dset_dict['lb_dset'][config["feature"]], 0, lb_indices)
        y_lb = torch.index_select(dset_dict['lb_dset']['label'], 0, lb_indices)
        x_ulb = torch.index_select(dset_dict['ulb_dset'][config["feature"]], 0, ulb_indices)
        x_tar = torch.index_select(dset_dict['tar_dset'][config["feature"]], 0, tar_indices)
        # predict
        pred_lb = student_model(x_lb)
        feature_lb = pred_lb["feature"]
        logits_lb = pred_lb["logits"]
        pred_ulb = student_model(x_ulb)
        feature_ulb = pred_ulb["feature"]
        pred_tar = student_model(x_tar)
        feature_tar = pred_tar["feature"]

        # Compute prediction error
        cls_loss = F.cross_entropy(logits_lb, y_lb)
        global_transfer_loss = global_domain_adv(torch.cat((feature_lb, feature_ulb)), torch.cat((feature_tar, feature_tar)))

        # predict pseudo label
        with torch.no_grad():
            teacher_pred_lb = teacher_model(x_lb)
            teacher_feature_lb = teacher_pred_lb["feature"]
            teacher_pred_ulb = teacher_model(x_ulb)
            teacher_feature_ulb = teacher_pred_ulb["feature"]
            teacher_pred_tar = teacher_model(x_tar)
            teacher_feature_tar = teacher_pred_tar["feature"]

            if len(feature_lb_queue) == queue_size:
                # 如果队列已满，则移除最旧的元素
                feature_lb_queue.popleft()
                feature_ulb_queue.popleft()
                feature_tar_queue.popleft()
                y_lb_queue.popleft()

            # 将当前 batch 的特征和标签加入队列
            feature_lb_queue.append(teacher_feature_lb)
            feature_ulb_queue.append(teacher_feature_ulb)
            feature_tar_queue.append(teacher_feature_tar)
            y_lb_queue.append(y_lb)

            # 从队列中获取所有的特征和标签
            all_teacher_feature_lb = torch.cat(list(feature_lb_queue), dim=0)
            all_teacher_feature_ulb = torch.cat(list(feature_ulb_queue), dim=0)
            all_teacher_feature_tar = torch.cat(list(feature_tar_queue), dim=0)
            all_y_lb = torch.cat(list(y_lb_queue), dim=0)

            # 通过标签传播获得伪标签
            pseudo_labels_ulb, pseudo_labels_tar = label_propagation(all_teacher_feature_lb, all_teacher_feature_ulb, all_teacher_feature_tar, all_y_lb, config["n_class"], config["sigma"], config["gamma"])
            pseudo_labels_ulb = sharpen(pseudo_labels_ulb[-len(teacher_feature_ulb):], config["t"])
            pseudo_labels_tar = sharpen(pseudo_labels_tar[-len(teacher_feature_tar):], config["t"])
        
        # 课程伪标签
        logits_ulb = pred_ulb["logits"]
        logits_tar = pred_tar["logits"]
        pseudo_labels_ulb = get_curriculum_pseudo_labels(config, logits_ulb, pseudo_labels_ulb, config["n_class"], config["threshold_min"], config["threshold_max"])
        pseudo_labels_tar = get_curriculum_pseudo_labels(config, logits_tar, pseudo_labels_tar, config["n_class"], config["threshold_min"], config["threshold_max"])

        sub_domain_transfer_loss = 0
        for j, sub_domain_adv in enumerate(sub_domain_advs):
            sub_domain_j_transfer_loss = sub_domain_adv(torch.cat((feature_lb, feature_ulb)), torch.cat((feature_tar, feature_tar)))
            weight = torch.cat((torch.eye(config["n_class"]).to(config["device"])[y_lb], pseudo_labels_ulb, pseudo_labels_tar, pseudo_labels_tar))[:, j]
            sub_domain_transfer_loss += (weight * torch.squeeze(sub_domain_j_transfer_loss, dim=1)).mean()

        global_domain_acc = global_domain_adv.domain_discriminator_accuracy

        # 计算 λ
        beta = config["beta"]
        tau = config["tau"]
        lamda = 1 / (1 + torch.exp(-beta * (torch.abs(global_domain_acc - 0.5) - tau)))

        transfer_loss = lamda * global_transfer_loss + (1 - lamda) * sub_domain_transfer_loss

        loss = cls_loss + transfer_loss * config["transfer_weight"]

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update teacher model
        update_teacher_model(student_model, teacher_model, config["alpha"])

def main_worker(sweep_q, worker_q):
    worker_data = worker_q.get()
    config = worker_data.config
    fold = worker_data.num + 1

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_id = fold % gpu_count
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    dset_dict = load_data(config, fold)

    student_model = Model(config).to(config["device"])
    teacher_model = Model(config).to(config["device"])
    teacher_model.load_state_dict(student_model.state_dict())
    domain_discri = DomainDiscriminator(in_feature=64, hidden_size=64).to(config["device"])
    optimizer = Adam(params=list(student_model.parameters()) + list(domain_discri.parameters()), lr=config["lr"])
    global_domain_adv = DomainAdversarialLoss(domain_discri).to(config["device"])

    sub_domain_advs = []
    params = list(student_model.parameters())
    for i in range(config["n_class"]):
        domain_discri = DomainDiscriminator(in_feature=64, hidden_size=64).to(config["device"])
        sub_domain_advs.append(DomainAdversarialLoss(domain_discri, 'none').to(config["device"]))
        params += list(domain_discri.parameters())

    best_accuracy = 0

    if fold == 1:
        print("This is the train process for fold1, and only fold1 is printed to simplify the output on the console.")

    for i in range(config["epochs"]):
        train_epoch(config, dset_dict, student_model, teacher_model, global_domain_adv, sub_domain_advs, optimizer)
        lb_loss, lb_accuracy = test(config, dset_dict['lb_dset'], student_model, config["feature"])
        ulb_loss, ulb_accuracy = test(config, dset_dict['ulb_dset'], student_model, config["feature"])
        test_loss, test_accuracy = test(config, dset_dict['tar_dset'], student_model, config["feature"])
        best_accuracy = max(best_accuracy, test_accuracy)
        domain_acc = global_domain_adv.domain_discriminator_accuracy
        if fold == 1:
            print(f"epoch{i + 1}: lb_loss={lb_loss:.4f}, lb_accuracy={lb_accuracy:.4f}, ulb_loss={ulb_loss:.4f}, ulb_accuracy={ulb_accuracy:.4f}, test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, best_accuracy={best_accuracy:.4f}, domain_acc={domain_acc:.4f}")
    sweep_q.put(WorkerDoneData(best_accuracy=best_accuracy))
    

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    default_config_SEED={
        # data
        "folds": 15,
        "dataset": "SEED",
        "num_subject": 15,
        "n_class": 3,
        "mode": "independ",
        # "n_labeled_trail": 9, # if mode == "depend"
        "n_labeled_subject": 13, # if mode == "independ"
        "feature": "de",
        "in_feature": 310,
        
        # alg
        "alg": "mgrada",
        "batch_size": 8,
        "epochs": 200,
        "it_per_epoch": 30,
        "optimizer": "Adam",
        "lr": 0.001,

        # specific
        "t": 0.4,               # sharpening temperature
        "transfer_weight": 4,   # weight of transfer loss
        "alpha": 0,          # momentum for teacher model
        "sigma": 0.1,          # bandwidth for similarity matrix
        "gamma": 0.99,          # decay for label propagation
        "beta": 0.3,             # weight of curriculum loss
        "tau": 0.6,             # threshold for curriculum loss
        "queue_size": 3,        # size of the queue for storing features
        "threshold_min": 0.5,   # minimum threshold for curriculum loss
        "threshold_max": 0.9,   # maximum threshold for curriculum loss
    }
    default_config_SEED_IV={
        # data
        "folds": 15,
        "dataset": "SEED-IV",
        "num_subject": 15,
        "n_class": 4,
        "mode": "independ",
        # "n_labeled_trail": 16, # if mode == "depend"
        "n_labeled_subject": 13, # if mode == "independ"
        "feature": "de",
        "in_feature": 310,

        # alg
        "alg": "mgrada",
        "batch_size": 8,
        "epochs": 200,
        "it_per_epoch": 30,
        "optimizer": "Adam",
        "lr": 0.001,

        # specific
        "t": 0.4,               # sharpening temperature
        "transfer_weight": 4,   # weight of transfer loss
        "alpha": 0,          # momentum for teacher model
        "sigma": 0.1,          # bandwidth for similarity matrix
        "gamma": 0.99,          # decay for label propagation
        "beta": 0.3,             # weight of curriculum loss
        "tau": 0.6,             # threshold for curriculum loss
        "queue_size": 3,        # size of the queue for storing features
        "threshold_min": 0.5,   # minimum threshold for curriculum loss
        "threshold_max": 0.9,   # maximum threshold for curriculum loss
    }
    default_config_FACED={
        # data
        "folds": 10,
        "dataset": "FACED",
        "num_subject": 123,
        "n_class": 9, # in [3, 9]
        "mode": "independ",
        "n_labeled_subject": 37, # if mode == "independ" in [37, 56, 74]
        "feature": "de",
        "in_feature": 160,

        # alg
        "alg": "mgrada",
        "batch_size": 32,
        "epochs": 10,
        "it_per_epoch": 50,
        "optimizer": "Adam",
        "lr": 0.0003,

        # specific
        "t": 0.9,               # sharpening temperature
        "transfer_weight": 1,   # weight of transfer loss
        "alpha": 0.99,          # momentum for teacher model
        "sigma": 0.25,          # bandwidth for similarity matrix
        "gamma": 0.99,          # decay for label propagation
        "beta": 10,             # weight of curriculum loss
        "tau": 0.5,             # threshold for curriculum loss
        "queue_size": 5,        # size of the queue for storing features
        "threshold_min": 0.5,   # minimum threshold for curriculum loss
        "threshold_max": 0.9,   # maximum threshold for curriculum loss
    }

    config = default_config_SEED
    
    config["device"] = "cpu"
    if torch.cuda.is_available():
        config["device"] = "cuda"
    # config["folds"] = 1
    paths = ["./data", "/kaggle/input/eeg-data", "/root/autodl-fs/eeg-data"]
    for path in paths:
        if os.path.exists(path):
            config["data_path"] = path
            break
    else:
        raise ValueError("No valid data path found.")
    main(config=config, main_worker=main_worker)
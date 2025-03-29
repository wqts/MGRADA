import os
import scipy.io as scio
import numpy as np
import pickle
import torch
from sklearn import preprocessing

SEED_session1_file_list = ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat', '4_20140621.mat', '5_20140411.mat', '6_20130712.mat', '7_20131027.mat', '8_20140511.mat', '9_20140620.mat', '10_20131130.mat', '11_20140618.mat', '12_20131127.mat', '13_20140527.mat', '14_20140601.mat', '15_20130709.mat']

SEED_session1_label = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]

SEED_IV_session1_file_list = ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat', '4_20151111.mat', '5_20160406.mat', '6_20150507.mat', '7_20150715.mat', '8_20151103.mat', '9_20151028.mat', '10_20151014.mat', '11_20150916.mat', '12_20150725.mat', '13_20151115.mat', '14_20151205.mat', '15_20150508.mat']

SEED_IV_session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]

SEED_IV_session2_file_list = ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat', '4_20151118.mat', '5_20160413.mat', '6_20150511.mat', '7_20150717.mat', '8_20151110.mat', '9_20151119.mat', '10_20151021.mat', '11_20150921.mat', '12_20150804.mat', '13_20151125.mat', '14_20151208.mat', '15_20150514.mat']

SEED_IV_session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]

SEED_IV_session3_file_list = ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat', '4_20151123.mat', '5_20160420.mat', '6_20150512.mat', '7_20150721.mat', '8_20151117.mat', '9_20151209.mat', '10_20151023.mat', '11_20151011.mat', '12_20150807.mat', '13_20161130.mat', '14_20151215.mat', '15_20150527.mat']

SEED_IV_session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def repermute(l: list, idx: list):
    assert(len(l) == len(idx))
    ans = []
    for i in idx:
        ans.append(l[i])
    return ans

def load_data(config, fold):
    if config["dataset"] == "SEED":
        if config["mode"] == "depend":
            dset_dict = load_SEED(fold, config["n_labeled_trail"], 1, config["data_path"])
        elif config["mode"] == "independ":
            dset_dict = load_SEED_independ(fold, config["n_labeled_subject"], 1, config["data_path"])
    elif config["dataset"] == "SEED-IV":
        if config["mode"] == "depend":
            dset_dict = load_SEED_IV(fold, config["n_labeled_trail"], 1, config["data_path"])
        elif config["mode"] == "independ":
            dset_dict = load_SEED_IV_independ(fold, config["n_labeled_subject"], 1, config["data_path"])
    elif config["dataset"] == "FACED":
        dset_dict = load_FACED_independ(fold, config["n_labeled_subject"], config["n_class"], config["data_path"])
    else:
        raise NameError(f"not find the dataset:{config.dataset}")
    transfer_all_data_to_tensor(dset_dict["lb_dset"])
    transfer_all_data_to_tensor(dset_dict["ulb_dset"])
    transfer_all_data_to_tensor(dset_dict["tar_dset"])
    load_all_data_to_cuda(dset_dict["lb_dset"])
    load_all_data_to_cuda(dset_dict["ulb_dset"])
    load_all_data_to_cuda(dset_dict["tar_dset"])
    return dset_dict


def load_SEED(subject, n_labeled_trail=3, session=1, data_path="./data"):
    assert(n_labeled_trail in [3, 6, 9])
    subject -= 1
    if session == 1:
        file_list = SEED_session1_file_list
        session_label = SEED_session1_label
    # elif session == 2:
    #     file_list = SEED_session2_file_list
    #     session_label = SEED_session2_label
    # elif session == 3:
    #     file_list = SEED_session3_file_list
    #     session_label = SEED_session3_label

    path = os.path.join(data_path, "SEED/ExtractedFeatures", file_list[subject])
    df = scio.loadmat(path)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    video_time=[235,233,206,238,185,195,237,216,265,237,235,233,235,238,206]

    de_list = []
    label_list = []

    for i in range(1, 16):
        de = np.transpose(df[f'de_LDS{i}'], (1, 0, 2)).reshape(-1, 310)
        psd = np.transpose(df[f'psd_LDS{i}'], (1, 0, 2)).reshape(-1, 310)
        label = np.full((len(de),), fill_value=session_label[i - 1])
        de_list.append(de)
        label_list.append(label)

    de = min_max_scaler.fit_transform(np.concatenate(de_list))
    label = np.concatenate(label_list)

    lb_num = sum(video_time[:n_labeled_trail])
    ulb_num = sum(video_time[:9])
    tar_num = sum(video_time[9:])

    lb_dset = {"de": de[:lb_num], "label": label[:lb_num]}
    ulb_dset = {"de": de[:ulb_num], "label": label[:ulb_num]}
    tar_dset = {"de": de[-tar_num:], "label": label[-tar_num:]}
    return {"lb_dset":lb_dset, "ulb_dset":ulb_dset, "tar_dset":tar_dset}

def load_SEED_independ(subject, n_labeled_subject=13, session=1, data_path="./data"):
    if session == 1:
        file_list = SEED_session1_file_list
        session_label = SEED_session1_label
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))

    lb_de_list = []
    lb_label_list = []
    ulb_de_list = []
    ulb_label_list = []
    tar_de = None
    tar_label = None
    for i in range(1, 16):
        path = os.path.join(data_path, "SEED/ExtractedFeatures", file_list[i - 1])
        df = scio.loadmat(path)
        de_list = []
        label_list = []
        for j in range(1, 16):
            de = np.transpose(df[f'de_LDS{j}'], (1, 0, 2)).reshape(-1, 310)
            label = np.full((len(de),), fill_value=session_label[j - 1])
            de_list.append(de)
            label_list.append(label)
        de = min_max_scaler.fit_transform(np.concatenate(de_list))
        label = np.concatenate(label_list)
        if i == subject:
            tar_de = de
            tar_label = label
        else:
            if len(lb_label_list) < n_labeled_subject:
                lb_de_list.append(de)
                lb_label_list.append(label)
            else:
                ulb_de_list.append(de)
                ulb_label_list.append(label)
    
    lb_de = np.concatenate(lb_de_list)
    lb_label = np.concatenate(lb_label_list)
    ulb_de = np.concatenate(ulb_de_list)
    ulb_label = np.concatenate(ulb_label_list)

    lb_dset = {"de": lb_de, "label": lb_label}
    ulb_dset = {"de": ulb_de, "label": ulb_label}
    tar_dset = {"de": tar_de, "label": tar_label}
    return {"lb_dset":lb_dset, "ulb_dset":ulb_dset, "tar_dset":tar_dset}


def load_SEED_IV(subject, n_labeled_trail=4, session=1, data_path="./data"):
    assert(n_labeled_trail in [4, 8, 12, 16])
    subject -= 1
    if session == 1:
        file_list = SEED_IV_session1_file_list
        session_label = SEED_IV_session1_label
    elif session == 2:
        file_list = SEED_IV_session2_file_list
        session_label = SEED_IV_session2_label
    elif session == 3:
        file_list = SEED_IV_session3_file_list
        session_label = SEED_IV_session3_label
    idx = argsort(session_label)
    tmp = []
    for i in range(6):
        for j in range(4):
            tmp.append(idx[i + j * 6])
    idx = tmp

    path = os.path.join(data_path, "SEED-IV/eeg_feature_smooth", str(session), file_list[subject])
    df = scio.loadmat(path)

    de_list = []
    label_list = []
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))
    video_time=[42,23,49,32,22,40,38,52,36,42,12,27,54,42,64,35,17,44,35,12,28,28,43,34]

    for i in range(1, 25):
        de = np.transpose(df[f'de_LDS{i}'], (1, 0, 2)).reshape(-1, 310)
        label = np.full((len(de),), fill_value=session_label[i - 1])
        de_list.append(de)
        label_list.append(label)

    video_time = repermute(video_time, idx)
    de_list = repermute(de_list, idx)
    label_list = repermute(label_list, idx)

    de = min_max_scaler.fit_transform(np.concatenate(de_list))
    label = np.concatenate(label_list)

    lb_num = sum(video_time[:n_labeled_trail])
    ulb_num = sum(video_time[:16])
    tar_num = sum(video_time[16:])

    lb_dset = {"de": de[:lb_num], "label": label[:lb_num]}
    ulb_dset = {"de": de[:ulb_num], "label": label[:ulb_num]}
    tar_dset = {"de": de[-tar_num:], "label": label[-tar_num:]}
    return {"lb_dset":lb_dset, "ulb_dset":ulb_dset, "tar_dset":tar_dset}


def load_SEED_IV_independ(subject, n_labeled_subject=13, session=1, data_path="./data"):
    if session == 1:
        file_list = SEED_IV_session1_file_list
        session_label = SEED_IV_session1_label
    elif session == 2:
        file_list = SEED_IV_session2_file_list
        session_label = SEED_IV_session2_label
    elif session == 3:
        file_list = SEED_IV_session3_file_list
        session_label = SEED_IV_session3_label
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))

    lb_de_list = []
    lb_label_list = []
    ulb_de_list = []
    ulb_label_list = []
    tar_de = None
    tar_label = None
    for i in range(1, 16):
        path = os.path.join(data_path, "SEED-IV/eeg_feature_smooth", str(session), file_list[i - 1])
        df = scio.loadmat(path)
        de_list = []
        label_list = []
        for j in range(1, 25):
            de = np.transpose(df[f'de_LDS{j}'], (1, 0, 2)).reshape(-1, 310)
            label = np.full((len(de),), fill_value=session_label[j - 1])
            de_list.append(de)
            label_list.append(label)
        de = min_max_scaler.fit_transform(np.concatenate(de_list))
        label = np.concatenate(label_list)
        if i == subject:
            tar_de = de
            tar_label = label
        else:
            if len(lb_label_list) < n_labeled_subject:
                lb_de_list.append(de)
                lb_label_list.append(label)
            else:
                ulb_de_list.append(de)
                ulb_label_list.append(label)
    
    lb_de = np.concatenate(lb_de_list)
    lb_label = np.concatenate(lb_label_list)
    ulb_de = np.concatenate(ulb_de_list)
    ulb_label = np.concatenate(ulb_label_list)

    lb_dset = {"de": lb_de, "label": lb_label}
    ulb_dset = {"de": ulb_de, "label": ulb_label}
    tar_dset = {"de": tar_de, "label": tar_label}
    return {"lb_dset":lb_dset, "ulb_dset":ulb_dset, "tar_dset":tar_dset}


def load_FACED_independ(fold, n_labeled_subject, n_class, data_path="./data"):
    assert(n_labeled_subject in [37, 56, 74])
    assert(n_class in [3, 9])
    path = os.path.join(data_path, "FACED/EEG_Features")
    min_max_scaler = preprocessing.MinMaxScaler(feature_range = (-1, 1))

    lb_de_list = []
    lb_label_list = []
    ulb_de_list = []
    ulb_label_list = []
    tar_de_list = []
    tar_label_list = []

    for i in range(123):
        with open(os.path.join(path, "DE", f"sub{str(i).zfill(3)}.pkl.pkl"), "rb") as f:
            de = pickle.load(f)
            de = np.transpose(de, (0, 2, 1, 3))
            de = de.reshape(840, 160)
        if n_class == 3:
            label = np.load(os.path.join(path, "label3", f"sub{str(i).zfill(3)}.npy"))
        else:
            label = np.load(os.path.join(path, "label9", f"sub{str(i).zfill(3)}.npy"))

        de = min_max_scaler.fit_transform(de)

        if i >= fold * 12 and i < (fold + 1) * 12:
            tar_de_list.append(de)
            tar_label_list.append(label)
        else:
            if len(lb_label_list) < n_labeled_subject:
                lb_de_list.append(de)
                lb_label_list.append(label)
            else:
                ulb_de_list.append(de)
                ulb_label_list.append(label)

    lb_de = np.concatenate(lb_de_list)
    lb_label = np.concatenate(lb_label_list)
    ulb_de = np.concatenate(ulb_de_list)
    ulb_label = np.concatenate(ulb_label_list)
    tar_de = np.concatenate(tar_de_list)
    tar_label = np.concatenate(tar_label_list)

    lb_dset = {"de": lb_de, "label": lb_label}
    ulb_dset = {"de": ulb_de, "label": ulb_label}
    tar_dset = {"de": tar_de, "label": tar_label}
    return {"lb_dset":lb_dset, "ulb_dset":ulb_dset, "tar_dset":tar_dset}


def transfer_all_data_to_tensor(dset: dict):
    dset["de"] = torch.from_numpy(dset["de"]).float()
    dset["label"] = torch.from_numpy(dset["label"]).long()

def load_all_data_to_cuda(dset: dict):
    dset["de"] = dset["de"].cuda().contiguous()
    dset["label"] = dset["label"].cuda().contiguous()

if __name__ == "__main__":
    # config = {"dataset": "SEED-IV", "n_labeled_trail": 4}
    # dset_dict = load_data(config, 1)
    load_FACED_independ(fold=1, n_labeled_subject=37, n_class=3)
    print(1)
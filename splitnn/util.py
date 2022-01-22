from torch.utils.data import Dataset
import torch
import pickle
import struct
import socket


def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msg_len = recvall(sock, 4)
    if not raw_msg_len:
        return None
    msg_len = struct.unpack('>I', raw_msg_len)[0]
    msg = recvall(sock, msg_len)
    msg = pickle.loads(msg)
    return msg


def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def data_loader(dataset, ratio, manual_seed):
    # split the dataset into train_set and test_set
    print("dataset: ", dataset)
    test_len = int(len(dataset) * ratio)
    total_len = int(len(dataset))
    train_len = total_len - test_len

    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len], torch.manual_seed(manual_seed))
    print("len(train_set):", len(train_set))
    print("len(test_set):", len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    print("len(train_loader):", len(train_loader))
    print("len(test_loader):", len(test_loader))
    return train_loader, test_loader


def agg_tensor(cut_parties, party_num):
    cut_aggregation = cut_parties[0]
    i = 1
    while i < party_num:
        cut_aggregation = cut_aggregation + cut_parties[i]
        i += 1
    return cut_aggregation

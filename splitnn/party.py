import torch
import time
import random
import socket
import argparse
from model import ServerSegment, PartySegment
from bank_dataset import BankDataset
from util import data_loader, recv_msg, send_msg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="ip address of the server")
    parser.add_argument("--server_port", type=int, default=5000, help="listening port of the server")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="ratio for splitting train/test dataset")
    parser.add_argument("--manual_seed", type=int, default=47, help="ip address of the server")
    parser.add_argument("--data_file", type=str, help="ip address of the server")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs in split learning")
    return parser.parse_args()


def connect_server(server_ip, server_port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # connect to server via web socket
        sock.connect((server_ip, server_port))
        return sock
    except socket.error as err:
        print("party socket connect server with error %s", err)


def check_test_accuracy(loader, party_model, sock):
    with torch.no_grad():
        for data, labels in loader:
            # forward computation
            cut_party = party_model.forward(data)
            # send output to server via socket
            try:
                send_msg(sock, cut_party)
            except socket.error as err:
                print("party send cut layer output error %s", err)


def party_train(train_loader, test_loader, party_model, sock, epochs):
    print("sock: ", sock)
    for epoch in range(epochs):
        for data, labels in train_loader:
            # zero grad
            party_model.zero_grads()
            # forward computation
            cut_party = party_model.forward(data)
            # send output to server via socket
            try:
                send_msg(sock, cut_party)
            except socket.error as err:
                print("party send cut layer output error %s", err)
            # receive gradient from server via socket
            server_grad = recv_msg(sock)
            # backward computation
            party_model.backward(server_grad)
            # update weights
            party_model.step()

        # assist to check test accuracy after each epoch's training
        check_test_accuracy(train_loader, party_model, sock)
        check_test_accuracy(test_loader, party_model, sock)
        print("Epoch {} accomplished".format(epoch))


if __name__ == '__main__':
    # parse arguments
    args = get_args()

    # set random seed for python and torch
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # init bank dataset and train/test data loader
    bank_party_set = BankDataset(args.data_file)
    train_loader, test_loader = data_loader(bank_party_set, args.split_ratio, args.manual_seed)

    # establish party connections
    sock = connect_server(args.server_ip, args.server_port)

    # create party model segment
    input_size = bank_party_set.balance_samples.shape[1]
    print("input_size = ", input_size)
    hidden_sizes = [32, 16]
    party_model = PartySegment(input_size, hidden_sizes)

    # start training
    start_time = time.time()
    print("\n---------- party_train started ----------\n")
    party_train(
        train_loader,
        test_loader,
        party_model,
        sock,
        args.epochs
    )
    print("\n---------- party_train finished ----------\n")
    end_time = time.time()
    print("party_train elapsed {} seconds\n".format(end_time - start_time))

    # close server connection
    sock.close()

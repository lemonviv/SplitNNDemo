import torch
import time
import random
import socket
import argparse
from model import ServerSegment, PartySegment
from bank_dataset import BankDataset
from util import data_loader, recv_msg, send_msg, agg_tensor
from plot import plot_train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", type=str, default="127.0.0.1", help="ip address of the server")
    parser.add_argument("--server_port", type=int, default=5000, help="listening port of the server")
    parser.add_argument("--party_num", type=int, default=2, help="number of parties in split learning")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs in split learning")
    parser.add_argument("--split_ratio", type=float, default=0.2, help="ratio for splitting train/test dataset")
    parser.add_argument("--manual_seed", type=int, default=47, help="ip address of the server")
    parser.add_argument("--data_file", type=str, help="ip address of the server")
    return parser.parse_args()


def listen_parties(server_ip, server_port, party_num):
    try:
        # create socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # waiting for connection from parties
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # bind and listen on server_ip and server_port
        s.bind((server_ip, server_port))
        s.listen()
        # accept connections from parties
        conn_array = []
        for i in range(party_num):
            conn, addr = s.accept()
            conn_array.append(conn)
            print('connected by', addr)

        return s, conn_array
    except socket.error as err:
        print("server socket with error %s", err)


def check_test_accuracy(loader, server_model, conn_array, party_num):
    correct = 0.0
    correct_base = 0.0
    with torch.no_grad():
        for data, labels in loader:
            # receive cut layer outputs from parties
            cut_parties = []
            try:
                for i in range(party_num):
                    cut_party_i = recv_msg(conn_array[i])
                    cut_parties.append(cut_party_i)
            except socket.error as err:
                print("server socket recv cut layer output with error %s", err)

            # aggregate cut layer outputs
            cut_aggregation = agg_tensor(cut_parties, party_num)

            # forward computation
            pred = server_model.forward(cut_aggregation)
            correct += ((torch.zeros_like(labels) + (pred > 0.5).squeeze()) == labels).sum()
            correct_base += data.shape[0]
        accuracy = correct / correct_base

    return correct, correct_base, accuracy


def server_train(train_loader, test_loader, server_model, conn_array, criterion, party_num, epochs):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        train_loss = 0.0
        for data, labels in train_loader:
            # zero grad
            server_model.zero_grads()

            # receive cut layer outputs from parties
            cut_parties = []
            try:
                for i in range(party_num):
                    cut_party_i = recv_msg(conn_array[i])
                    cut_parties.append(cut_party_i)
            except socket.error as err:
                print("server socket recv cut layer output with error %s", err)

            # aggregate cut layer outputs
            cut_aggregation = agg_tensor(cut_parties, party_num)

            # forward computation
            pred = server_model.forward(cut_aggregation)

            # compute loss
            labels = labels.type(torch.FloatTensor)
            loss = criterion(pred, labels)

            # backward computation
            loss.backward()
            server_grad = server_model.backward()

            # send gradients to parties
            for i in range(party_num):
                send_msg(conn_array[i], server_grad)

            # update weights
            server_model.step()

            # record loss
            train_loss += loss

        train_loss = train_loss / len(train_loader)
        # evaluate on the train dataset
        train_correct, train_correct_base, train_accuracy = check_test_accuracy(
           train_loader, server_model, conn_array, party_num)
        # evaluate on the test dataset
        test_correct, test_correct_base, test_accuracy = check_test_accuracy(
            test_loader, server_model, conn_array, party_num)
        # append the train_loss, train_accuracy, test_accuracy
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print("Epoch {}, Training loss {}, "
              "Training accuracy {}/{} ({}%), "
              "Testing accuracy {}/{} ({}%)".format(epoch, train_loss,
                                                    train_correct, train_correct_base, train_accuracy,
                                                    test_correct, test_correct_base, test_accuracy))

    return train_losses, train_accuracies, test_accuracies


if __name__ == '__main__':
    # parse arguments
    args = get_args()

    # set random seed for python and torch
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    # init bank dataset and train/test data loader
    bank_server_set = BankDataset(args.data_file)
    train_loader, test_loader = data_loader(bank_server_set, args.split_ratio, args.manual_seed)

    # establish party connections
    s, conn_array = listen_parties(args.server_ip, args.server_port, args.party_num)

    # create server model segment
    hidden_sizes = [16, 1]
    server_model = ServerSegment(hidden_sizes)
    criterion = torch.nn.BCELoss()

    # start training
    start_time = time.time()
    print("\n---------- server_train started ----------\n")
    train_losses, train_accuracies, test_accuracies = server_train(
        train_loader,
        test_loader,
        server_model,
        conn_array,
        criterion,
        args.party_num,
        args.epochs
    )
    print("\n---------- server_train finished ----------\n")
    end_time = time.time()
    print("server_train elapsed {} seconds\n".format(end_time - start_time))

    # plot loss, train_accuracy, test_accuracy figure
    plot_train(train_losses, train_accuracies, test_accuracies)

    # close party connections
    s.close()

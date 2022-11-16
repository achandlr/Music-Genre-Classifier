# import the models
from models.M5_Audio_Classifier import *
from models.FeedForward import *

from utils.dataset import AudioDataset
from torch.utils.data import Dataset, DataLoader
from utils.parsers import training_parser
import numpy as np
from tqdm.notebook import tqdm
import torch.nn.functional as F
from collections import deque
import pickle # TODO: can replace with h5py file 


def get_data_loaders(dataset, batch_size):
    # 60% - train set, 20% - validation set, 20% - test set
    train_indices, validate_indices, test_indices = np.split(np.arange(len(dataset)), [int(.6*len(dataset)), int(.8*len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  sampler=train_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  sampler=validate_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_indices)
    return train_loader, val_loader, test_loader

# heavily modified from existing implementation of a computer vision training loop I built: https://github.com/achandlr/Musical-Instruments/blob/master/2022%20Implementation%20(Improved%20Implementation%20With%20Different%20Focus)/Using%20Transfer%20Learning%20for%20Musical%20Instrument%20Classification.ipynb  
def test_network(model, test_loader, description, debug= False, device = "cpu"):
    correct = 0
    total = 0
    true, pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels  in test_loader:
          inputs = inputs.to(device)
          labels = labels.to(device)
          outputs = model.forward(inputs)
          outputs = torch.squeeze(outputs)
          predicted = torch.argmax(outputs.cpu(), dim=1)
          total += labels.size(0)
          correct += (predicted == labels.cpu()).sum().item()
          true.append(labels)
          pred.append(predicted)
          if debug and total>100:
              break       
    acc = (100 * correct / total)
    print('%s has a test accuracy of : %0.3f' % (description, acc))
    return acc

# heavily modified from existing implementation of a computer vision training loop I built: https://github.com/achandlr/Musical-Instruments/blob/master/2022%20Implementation%20(Improved%20Implementation%20With%20Different%20Focus)/Using%20Transfer%20Learning%20for%20Musical%20Instrument%20Classification.ipynb  
def train_network_with_validation(model, train_loader, val_loader, test_loader, criterion, optimizer, description, num_epochs=20, device = "cpu"):
    queue_capacity=1000
    loss_queue = deque(maxlen=queue_capacity)
    queue_loss_list = []
    train_loss_list = []
    val_loss_list = []
    try:
        for epoch in tqdm(range(num_epochs)):
            problem_cnt = 0
            model.train()
            print('EPOCH %d'%epoch)
            total_loss = 0
            count = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                # print(inputs.shape)
                labels = labels.to(device)
                optimizer.zero_grad()
                # print(inputs.shape)
                outputs = model.forward(inputs)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels) 
                # print("loss {}".format(loss))
                loss_queue.append(loss.item())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            # print("problem_cnt: {}".format(problem_cnt))
            train_loss = total_loss/count
            train_loss_list.append(train_loss)
            print('{:>12s} {:>7.5f}'.format('Train loss:', train_loss))
            with torch.no_grad():
                total_loss = 0
                count = 0
                for inputs, labels in val_loader:
                  inputs = inputs.to(device)
                  labels = labels.to(device)
                  outputs = model.forward(inputs)
                  outputs = torch.squeeze(outputs)
                  loss = criterion(outputs, labels)
                  total_loss += loss.item()
                  count += 1
                val_loss = total_loss/count
                print('{:>12s} {:>7.5f}'.format('Val loss:', val_loss))
                val_loss_list.append(val_loss)
            print()
    except KeyboardInterrupt:
        print('Exiting from training early')
    return queue_loss_list, train_loss_list, val_loss_list


# Note to grader: K-Fold validation and other cross validation techniques are not used due to computational constraints
if __name__ == "__main__":
    args = training_parser.parse_args()
    args.batch_size = 4 # TODO: delete later
    args.device = "cpu"
    args.num_epochs = 5
    args.model_name = "M5"
    args.audio_folder_path = "data/fma_small"
    args.sampling_freq = 500_000 
    args.padding_length = None
    args.truncation_length = None #1300000
    args.convert_one_channel = True
    # args.load_dataset_path = "logs/datasets/dataset_fma_small_one_channel"
    args.load_dataset_path = None
    args.debug = True  # TODO delete
    args.desired_dataset_name = "dataset_fma_small_one_channel_torch_4k_samples500_000"
    args.datatype = "torch"
    if args.audio_folder_path == "data/fma_small":
        num_genres = 8
    else:
        raise NotImplementedError()
    # build preprocessing_dict from arg parameters
    preprocessing_dict = {
        "sampling_freq": args.sampling_freq,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }
    num_genres = 8
    if args.model_name == "M5":
        n_input = 1_300_000 # TODO: likely need to change
        if args.convert_one_channel:
            in_channels = 1
        else:
            in_channels = 2
        model = M5(n_input=in_channels, n_output=num_genres) 
        lr = 1e-4
        # can also experiment with different parameters
        optimizer = optim.Adam(model.parameters(), lr=lr) 
        # can also try other optimzers like SGD
        epochs = 30
        criterion = nn.CrossEntropyLoss()
        description = "Training M5 CNN model with Adam and CrossEntropyLoss"
        test_description = "Testing M5 CNN model on test data"
    
    elif args.model_name == "FF":
        n_input = 1_300_000
        num_genres = 8
        model = FF(n_input=n_input, n_output=num_genres)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 50
        criterion = nn.SGD()
        description = "Training FF model with Adam and SGD"
        test_description = "Testing FF model on test data"
    else:
        raise NotImplementedError("Model name not implemented")
    if args.load_dataset_path != None:
        with open(args.load_dataset_path, "rb") as input_file:     
            dataset = pickle.load(input_file)
    else:
        dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug, datatype = args.datatype)
        print("poop")
        # save dataset in logs/datasets
        with open("logs/datasets/"+args.desired_dataset_name, "wb") as output_file:
            pickle.dump(dataset, output_file)
    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=args.batch_size)

    queue_loss_list, train_loss_list, val_loss_list = train_network_with_validation(model, train_loader, val_loader, test_loader, criterion, optimizer, description, num_epochs=args.num_epochs, device = "cpu")
    test_acc = test_network(model, test_loader, description)
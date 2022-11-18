# import the models
from models.M5_Audio_Classifier import *
from models.FeedForward import *
from models.CNN_Custom_1 import *
from utils.dataset import AudioDataset
from torch.utils.data import Dataset, DataLoader
from utils.parsers import training_parser
import numpy as np
from tqdm.notebook import tqdm
import torch.nn.functional as F
from collections import deque
import pickle # Could replace with h5py file for security reasons
import datasets

def get_data_loaders(dataset, batch_size):
    # 60% - train set, 20% - validation set, 20% - test set
    train_indices, validate_indices, test_indices = np.split(np.arange(len(dataset)), [int(.6*len(dataset)), int(.8*len(dataset))])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  sampler=train_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  sampler=validate_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_indices)
    return train_loader, val_loader, test_loader

# heavily modified from existing implementation of a computer vision training loop Alex built: https://github.com/achandlr/Musical-Instruments/blob/master/2022%20Implementation%20(Improved%20Implementation%20With%20Different%20Focus)/Using%20Transfer%20Learning%20for%20Musical%20Instrument%20Classification.ipynb  
def test_network(model, test_loader, description, debug= False, device = "cpu", topk=1):
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
            if topk == 1:
                # unbatched case
                if len(outputs.size()) ==1:
                    predicted = torch.argmax(outputs.cpu())
                else:
                    predicted = torch.argmax(outputs.cpu(), dim=1)
                correct += (predicted == labels.cpu()).sum().item()
            else:
                predicted =torch.topk(outputs, topk).indices
                for idx, label in enumerate(labels):
                    if label in predicted[idx]:
                        correct +=1
                # top_k_predictions = torch.topk(outputs.cpu(), topk)
            total += labels.size(0)
            true.append(labels)
            pred.append(predicted)   
    acc = (100 * correct / total)
    print('%s has a test accuracy of : %0.3f' % (description, acc))
    return acc

# heavily modified from existing implementation of a computer vision training loop I built: https://github.com/achandlr/Musical-Instruments/blob/master/2022%20Implementation%20(Improved%20Implementation%20With%20Different%20Focus)/Using%20Transfer%20Learning%20for%20Musical%20Instrument%20Classification.ipynb  
def train_network_with_validation(model, train_loader, val_loader, test_loader, criterion, optimizer, description, num_epochs=20, device = "cpu", scheduler = None,  batch_size = 1):
    queue_capacity=1000
    loss_queue = deque(maxlen=queue_capacity)
    queue_loss_list = []
    train_loss_list = []
    val_loss_list = []
    try:
        for epoch in tqdm(range(num_epochs)):
            # problem_cnt = 0
            model.train()
            print('EPOCH %d'%epoch)
            total_loss = 0
            count = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                # assert inputs.requires_grad==True 
                # print(inputs.shape)
                labels = labels.to(device)
                optimizer.zero_grad()
                # print(inputs.shape)
                outputs = model.forward(inputs)
                if batch_size>1:
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
                  if batch_size>1:
                    outputs = torch.squeeze(outputs)
                  loss = criterion(outputs, labels)
                  total_loss += loss.item()
                  count += 1
                val_loss = total_loss/count
                print('{:>12s} {:>7.5f}'.format('Val loss:', val_loss))
                val_loss_list.append(val_loss)
            if scheduler:
                scheduler.step()
            print()
    except KeyboardInterrupt:
        print('Exiting from training early')
    return queue_loss_list, train_loss_list, val_loss_list


# Note to grader: K-Fold validation and other cross validation techniques are not used due to computational constraints
if __name__ == "__main__":
    args = training_parser.parse_args()
    args.batch_size = 32 
    args.device = "cpu"
    args.num_epochs = 5
    args.model_name = "wav2vec"
    args.audio_folder_path = "data/fma_small" 
    args.sampling_freq = 16_000 # 8_000
    args.truncation_length =   120_000 #1300000 400_000 None
    args.padding_length = None 
    args.convert_one_channel = True 
    args.dump_dataset = False # TODO: insert to arg parser
    args.save_model_path = "logs/models/" + args.model_name
    args.load_model_path = None # "logs/models/" + args.model_name
    args.load_dataset_path = None # "logs/datasets/dataset_fma_small_one_channel_torch_4k_samples500_000" # "logs/datasets/dataset_fma_small_one_channel_datatypetorch_samples200_truncation200_000_sampling8_000"
    args.debug = True  # TODO delete
    args.desired_dataset_name = "dataset_fma_small_one_channel_datatypetorch_samples200_truncation200_000_sampling8_000"
    args.datatype = "torch"
    args.acceptable_genres = None # ['Hip-Hop', 'Jazz']
    args.topk = 3 # default is 1
    if args.audio_folder_path == "data/fma_small":
        num_genres = 8
    else:
        raise NotImplementedError("Limiting scope of project to 8,000 audio files due to computational and memory constraints.")
    # build preprocessing_dict from arg parameters
    preprocessing_dict = {
        "sampling_freq": args.sampling_freq,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }
    if args.convert_one_channel:
            in_channels = 1
    else:
        in_channels = 2
    scheduler = None
    if args.model_name == "M5":
        model = M5(n_input=in_channels, n_output=num_genres) 
        lr = 1e-3
        # can also experiment with different parameters
        optimizer = optim.Adam(model.parameters(), lr=lr) # torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        description = "Training M5 CNN model with Adam and CrossEntropyLoss"
        test_description = "Testing M5 CNN model on test data"
    elif args.model_name == "FF":
        model = FF(n_input=in_channels, n_output=num_genres)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.SGD()
        description = "Training FF model with Adam and SGD"
        test_description = "Testing FF model on test data"
    elif args.model_name == "CNN_Custom1":
        model = CNN_Custom1(n_input=in_channels, n_output=num_genres)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) 
        criterion = nn.CrossEntropyLoss()
        description = "Training CNN_Custom1 model with Adam and CrossEntropyLoss"
        test_description = "Testing CNN_Custom1 CNN model on test data"
    # Use SOTA Audio Classifier from the huggingface library
    elif args.model_name == "wav2vec":
        from transformers import AutoFeatureExtractor
        from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
        from datasets import load_dataset, load_metric
        from torch.utils.data import random_split
        metric = load_metric("accuracy")
        # Note to grader: the following is somewhat out of scope of the project, and relies on modified starter code from from https://huggingface.co/docs/transformers/tasks/audio_classification
        torch_dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug, datatype = args.datatype, return_type = dict)
        train_dataset, val_dataset, test_dataset, = random_split(torch_dataset, [.6, .2, .2]) # maybe need .values()
        # build a DatasetDict of the split datasets
        dataset = datasets.DatasetDict({"train":train_dataset,"validation": val_dataset, "test":test_dataset})
        num_labels = len(torch_dataset.genres_factorized[1])
        # create a one to one mapping and inverse mapping of id2label
        id2label,label2id = {}, {}
        for id in range(num_labels):
            id2label[str(id)] = torch_dataset.genres_factorized[1][id]
            label2id[torch_dataset.genres_factorized[1][id]] = str(id)
        # print(id2label)
        # print(label2id)

        # Note to grader: The following code is modified from a colab notebook linked in the official huggingface website (linked below)
        # https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb#scrollTo=EVWfiBuv2uCS
        def compute_metrics(eval_pred):
            """Computes accuracy on a batch of predictions"""
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return metric.compute(predictions=predictions, references=eval_pred.label_ids)
        # instantiates on a pre trained audio classifcation model
        model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label)
        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-5,
            num_train_epochs=args.num_epochs,
        )
        model_checkpoint = "./logs/models/wav2vec2-base" # TODO: change to 
        model_name = model_checkpoint.split("/")[-1]
        training_args = TrainingArguments(
        f"{model_name}-finetuned-ks",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # push_to_hub=True,
        )
        trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics)
        trainer.train()
        trainer.evaluate()
        exit()
    else:
        raise NotImplementedError("Model name not implemented")
    if args.load_dataset_path != None:
        with open(args.load_dataset_path, "rb") as input_file:     
            dataset = pickle.load(input_file)
    else:
        dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug, datatype = args.datatype)
        # save dataset in logs/datasets
        if args.dump_dataset:
            with open("logs/datasets/"+args.desired_dataset_name, "wb") as output_file:
                pickle.dump(dataset, output_file)
    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size=args.batch_size)
    if args.load_model_path!=None:
        model.load_state_dict(torch.load(args.load_model_path))
        # model.eval()
    # transformers are trained differently than a CNN so ignore that for now
    # if args.model_name != "wav2vec":
    queue_loss_list, train_loss_list, val_loss_list = train_network_with_validation(model, train_loader, val_loader, test_loader, criterion, optimizer, description, num_epochs=args.num_epochs, device = "cpu", scheduler = scheduler, batch_size = args.batch_size)
    test_acc = test_network(model, test_loader, description, topk = args.topk)
    if args.save_model_path!= None:
        torch.save(model.state_dict(), args.save_model_path)
        # model.eval()
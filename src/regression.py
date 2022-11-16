from sklearn.model_selection import cross_val_score
from utils.dataset import AudioDataset
import sklearn.metrics as metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

args = training_parser.parse_args()
args.batch_size = 4 # TODO: delete later
args.device = "cpu"
args.num_epochs = 5
args.model_name = "M5"
args.audio_folder_path = "data/fma_small"
args.sampling_freq = 10000
args.padding_length = None
args.truncation_length = 10000 # TODO_modify
args.convert_one_channel = True
args.load_dataset_path = None # or logs/datasets/dataset_fma_small_one_channel
args.debug = True  # TODO delete
args.desired_dataset_name = "dataset_fma_small_one_channel_np"
args.datatype = "np"
if args.audio_folder_path == "data/fma_small":
    num_genres = 8
else:
    raise NotImplementedError()

preprocessing_dict = {
        "sampling_freq": args.sampling_freq,
        "padding_length": args.padding_length,
        "truncation_len" : args.truncation_length,
        "convert_one_channel": args.convert_one_channel
    }

USE_GRID_SEARCH = False
dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)


x_train, x_test, y_train, y_test = train_test_split(dataset.audio_tensors, dataset.genres_factorized[0], test_size=0.2, random_state=24)
# create a linear regression object
regression = LinearRegression().fit(x_train, y_train)

# predict the response for test dataset
y_pred = regression.predict(x_test)

# accuracy important?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# TODO: should we do more than MSE like the hw's?
print("Mean squared error: %.2f"
        % metrics.mean_squared_error(y_test, y_pred))

# TODO: should we visualize here? 
# # visualize
# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred, color="black", linewidth=3)

# # 5 fold cross validation on regression model
# cvlr = cross_val_score(LinearRegression(), x, y, cv=5)
# # print each cv score (accuracy) and average them
# print("Average 5-Fold CV Score: {}".format(np.mean(cvlr)))

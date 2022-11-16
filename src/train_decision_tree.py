from utils.dataset import AudioDataset
from sklearn import tree

args = CNNTrainingParser.parse_args()


dataset = AudioDataset(meta_data_path = "data/fma_metadata", audio_folder_path = args.audio_folder_path, preprocessing_dict = preprocessing_dict, debug = args.debug)


decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree = decision_tree.fit(x_train, y_train)
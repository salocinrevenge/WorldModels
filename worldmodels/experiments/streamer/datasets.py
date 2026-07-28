import os
import cv2
import torch
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from torchvision import transforms as T

class Epic_dataset:

    def __init__(self, data_path: str, load_once = False, snippet_size=0.5, resize = (128,128)):
        """
        :param data_path: Path to the folder containing tree of all video frames
        :type data_path: str
        :param ground_truths_path: Path to the tree of ground truth files for the dataset
        :param load_once: If True, loads all data into memory at once. Default is False.
        :type load_once: bool
        :param snippet_size: Time duration (in seconds) represented by each frame snippet. Default is 0.5 seconds.
        :type snippet_size: float
        
        example: ds = Epic_dataset('../datasets/epic_kitchens/frames', load_once=False)
        """
        paths = []
        ground_truths_paths = []
        ground_truths_path = os.path.join(data_path, '../groundTruth/')
        # Verificar se a pasta possui subpastas de participantes (P01, P02, etc.). Para essas pastas com esse formato,
        # acessar as subpastas dentro de videos/ e verificar se elas possuem o formato P01_01, P01_02, etc. Para cada pasta com esse formato,
        #  adicionar esse caminho da pasta de frames correspondente em paths. Exemplo: 'datasets/epic_kitchens/EPIC-KICTHENS/frames/P01/videos/P01_01'
        for participant in sorted(os.listdir(data_path)): # 'P01' or "map.json"
            if participant.startswith('P') and participant[1:].isdigit(): # only P01, P02, etc
                participant_path = os.path.join(data_path, participant) # 'datasets/epic_kitchens/EPIC-KICTHENS/frames/P01'
                videos_path = os.path.join(participant_path, 'videos')
                participant_gt_path = os.path.join(ground_truths_path, participant)
                if os.path.exists(videos_path) and os.path.exists(participant_gt_path):  # check if paths exist
                    for video in sorted(os.listdir(videos_path)):
                        if video.startswith(participant + '_') and video[len(participant)+1:].isdigit():
                            video_path = os.path.join(videos_path, video)
                            gt_file_path = os.path.join(participant_gt_path, video + '.json')
                            if os.path.exists(video_path) and os.path.exists(gt_file_path) and len(os.listdir(video_path)) > 0  :
                                paths.append(video_path)
                                ground_truths_paths.append(gt_file_path)
        
            
        self.paths = paths
        self.ground_truths_paths = ground_truths_paths
        assert len(self.paths) == len(self.ground_truths_paths), "Number of video paths and ground truth paths must be the same"
        self.load_once = load_once
        self.snippet_size = snippet_size
        self.resize = resize
        self.transform = T.Compose([
                    T.ToTensor(),
                    T.Resize(resize, antialias=True),
                ])
        if self.load_once:
            self.data = []
            self.gt = []
            self.jsons = []
            for path in self.paths:
                frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
                video_frames = []

                for img_path in frames:
                    img = cv2.imread(img_path)
                    assert (type(resize) == tuple) and (len(resize) == 2), "Resize must be a tuple (width, height)"
                    img = cv2.resize(img, resize)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames = (self.transform(img)*2.0-1.0).unsqueeze(0)

                    # img_tensor = torch.from_numpy(img).float()
                    # img_tensor = img_tensor.permute(2, 0, 1)
                    # img_tensor = img_tensor.unsqueeze(0)
                    # img_tensor = img_tensor / 255.0

                    # video_frames.append(img_tensor)
                    video_frames.append(frames)
                
                self.data.append(video_frames)
                self.jsons.append(json.load(open(self.ground_truths_paths[len(self.gt)], 'r')))
                self.gt.append(self.gt_json_to_sequence(self.ground_truths_paths[len(self.gt)], snippet_size, len(self.data[-1])))
            self.data = [torch.cat(video_frames, dim=0) for video_frames in self.data]

    def gt_json_to_sequence(self, gt_path, snippet_size, target_size):
        with open(gt_path, 'r') as f:
            data = json.load(f)
                
            # Get annotations from the JSON
            annots = sorted(data['layers'][0]['annots'], key=lambda x: x['start'])
            duration = data['duration']
            
            # Calculate number of snippets
            num_snippets = int(np.ceil(duration / snippet_size))
            
            # Initialize sequence with -1 (no action)
            sequence = np.full(num_snippets, -1, dtype=np.int32)
    
            current_id = 0
            last_action = None
            last_end = 0
            for annot in annots:
                start = annot['start']
                end = annot['end']
                action_name = str(annot['action'])
                if last_action != action_name and "still " not in action_name:
                    last_end = end
                    current_id += 1
                    last_action = action_name
                else:
                    start = last_end

                

                # Convert start and end times to snippet indices
                start_idx = int(start / snippet_size)
                end_idx = int(end / snippet_size)
                
                # Fill sequence with action ID
                sequence[start_idx:end_idx + 1] = current_id
            
            # Fill gaps new ids
            new = True
            for i in range(num_snippets):
                if sequence[i] == -1:
                    if new:
                        current_id += 1
                    sequence[i] = current_id
                    new = False
                    if i< num_snippets - 1 and sequence[i+1] != -1:
                        new = True


            # Adjust sequence length to match target_size
            if len(sequence) > target_size:
                sequence = sequence[:target_size]
            elif len(sequence) < target_size:
                last_id = sequence[-1]
                padding = np.full(target_size - len(sequence), last_id, dtype=np.int32)
                sequence = np.concatenate([sequence, padding])
            
            return sequence
        

    def __len__(self):
        return len(self.paths)
    

    def get_json(self, idx):
        if not self.load_once:
            return json.load(open(self.ground_truths_paths[idx], 'r'))
        else:
            return self.jsons[idx]

    def __getitem__(self, idx):
        if not self.load_once:
            path = self.paths[idx]
            print(f"Loading frames from path: {path}")
            frames = sorted(glob.glob(os.path.join(path, '*.jpg')))
            video_frames = []

            for img_path in frames:
                img = cv2.imread(img_path)
                if self.resize is not None:
                    assert (type(self.resize) == tuple) and (len(self.resize) == 2), "Resize must be a tuple (width, height)"
                    img = cv2.resize(img, self.resize)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_tensor = torch.from_numpy(img).float()
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = img_tensor.unsqueeze(0)
                img_tensor = img_tensor / 255.0

                video_frames.append(img_tensor)

            if len(video_frames) == 0:
                return torch.tensor([]), np.array([])
            video_frames = torch.cat(video_frames, dim=0)
            return video_frames, self.gt_json_to_sequence(self.ground_truths_paths[idx], self.snippet_size, video_frames.shape[0])
        else:
            return self.data[idx], self.gt[idx]
    
class BreakfastDataset:
    def __init__(self, dataset_path, load_once = False, d_type = "float64", percentage = 1):
        self.d_type = d_type
        self.path_ds = os.path.join(dataset_path, "Breakfast")
        self.path_gt = os.path.join(self.path_ds, 'groundTruth/')
        self.percentage = percentage
        path_mapping = os.path.join(self.path_ds, 'mapping', 'mapping.txt')
        path_features = os.path.join(self.path_ds, 'features/')
        # Load all needed files: Descriptor, GT & Mapping
        if os.path.exists(path_mapping):
            # Create the Mapping dict
            self.mapping_dict = self.get_mapping(path_mapping)
        else:
            self.mapping_dict = None
        

        # Load all filenames from ds path        
        self.filenames = sorted(glob.glob(os.path.join(path_features, '**/*.txt'), recursive=True))
        self.length = len(self.filenames)
        self.load_once = load_once

        if self.load_once:
            self.n_clusters = []
            self.actions = []
            self.features = []
            self.labels = []

            for idx in tqdm(range(self.length), desc="Loading Breakfast Dataset"):
                features, gt_labels, n_clusters, activity_name = self.load_item(idx)
                self.features.append(features)
                self.labels.append(gt_labels)
                self.n_clusters.append(n_clusters)
                self.actions.append(activity_name)
    
    def get_action(self, idx):
        if self.load_once:
            return self.actions[idx]
        else:
            _, _, _, activity_name = self.load_item(idx)
            return activity_name

    def get_paths_gt_paths(self):
        return self.filenames, [os.path.join(self.path_gt, os.path.basename(f).replace('.txt', '')) for f in self.filenames]

    def get_mean_number_of_episodes(self):
        if self.load_once:
            return int(np.mean(self.n_clusters))
        else:
            return 5

    def get_mapping(self, mapping_file_path):
        df = pd.read_csv(mapping_file_path, sep=" ", header=None, names=['index', 'action_name'])
        mapping_dict = dict(zip(df['action_name'], df['index']))
        return mapping_dict

    def get_n_clusters(self, idx):
        return self.n_clusters[idx]

    def load_gt_label(self, idx, mapping_dict=None):
        video_name = os.path.basename(self.filenames[idx])[:-4]
        gt_label_path = os.path.join(self.path_gt, video_name)
        gt_labels, n_labels = self.read_gt_label(gt_label_path, mapping_dict=self.mapping_dict)
        return gt_labels
        
    def load_item(self, idx):
        cur_desc = np.loadtxt(self.filenames[idx], dtype=self.d_type)

        # Load the GT_labels, map them to the corresponding ID    
        video_name = os.path.basename(self.filenames[idx])[:-4]
        activity_name = os.path.basename(os.path.split(self.filenames[idx])[0])

        # n_clusters for TWFINCH
        n_clusters = int(self.avg_gt_activity_datasets["Breakfast"][activity_name])
        gt_label_path = os.path.join(self.path_gt, video_name)
        if not os.path.exists(gt_label_path):
            gt_label_path = os.path.join(self.path_gt, video_name + '.txt')

        gt_labels, n_labels = self.read_gt_label(gt_label_path, mapping_dict=self.mapping_dict)

        return cur_desc, gt_labels, n_clusters, activity_name

    avg_gt_activity_datasets = {'Breakfast': {'cereals': 5, 'coffee': 7, 'friedegg': 9, 'juice': 8,
                                          'milk': 5, 'pancake': 14, 'salat': 8,
                                          'sandwich': 9, 'scrambledegg': 12, 'tea': 7}}
    avg_gt_activity_datasets_twfinch = {'Breakfast': {'cereals': 4, 'coffee': 4, 'friedegg': 6, 'juice': 5,
                                          'milk': 4, 'pancake': 9, 'salat': 5,
                                          'sandwich': 5, 'scrambledegg': 7, 'tea': 4}}
    
    def read_gt_label(self, gt_label_path, mapping_dict=None):
        df_gt = pd.read_csv(gt_label_path, sep=" ", header=None)
        gt = df_gt[0].tolist()
        if mapping_dict is not None:
            gt_label = [mapping_dict[i] for i in gt]
            gt_label = np.array(gt_label)
            n_labels = len(mapping_dict)
        else:
            _, gt_label = np.unique(gt, return_inverse=True)
            n_labels = len(np.unique(gt_label))

        # make sure gt label do not contain -ve entries
        gt_min = np.min(gt_label)
        if gt_min < 0:
            gt_label = gt_label - gt_min

        return gt_label, n_labels
    
    def __len__(self):
        return self.length*self.percentage
    
    def __getitem__(self, idx):
        if self.load_once:
            return self.features[idx], self.labels[idx]
        else:
            features, gt_labels, n_clusters, activity_name = self.load_item(idx)
            return features, gt_labels

class CarDataset:
    def __init__(self, paths: list, load_once = False, window_size=300, step_size=300, normalization=None):
        """
        Dataset for car trajectory data stored in CSV files.
        Args:
            paths (list): List of file paths to the CSV files.
            load_once (bool): If True, loads all data into memory at once.
        Example:
            ds = CarDataset(["../dataset/car/domingo/domingo.csv", "../dataset/car/segunda/segunda.csv", "../dataset/car/terca/terca.csv", "../dataset/car/delivery/delivery.csv", "../dataset/car/pizza/pizza.csv"], load_once=True)
        """
        self.paths = sorted(paths)
        self.load_once = load_once
        self.window_size = window_size
        self.step_size = step_size
        self.normalization = normalization
        if self.load_once:
            self.data = []
            self.features = []
            self.labels = []
            self.n_clusters = []

            for path in self.paths:
                vector = self.load_data_car(path, columns_to_import=list("timestamp,latitude,longitude,speed,speed_x,speed_y,acceleration,acceleration_x,acceleration_y,angle,acc_diff,desc".split(",")))

                vector[:,-1], numero_clusters = self.translate_labels(vector[:,-1])
                self.n_clusters.append(numero_clusters)
                self.data.append(vector)

                features, labels = self.window(vector, self.window_size, self.step_size)
                if normalization == "standard":
                    mean = np.mean(features, axis=0, keepdims=True)
                    std = np.std(features, axis=0, keepdims=True) + 1e-8
                    features = (features - mean) / std

                self.features.append(features)
                self.labels.append(labels)

    def get_mean_number_of_episodes(self):
        if self.load_once:
            return int(np.mean(self.n_clusters))
        else:
            return 33

    def window(self, data, window_size, step_size):
        num_windows = data.shape[0] // step_size

        features = []
        labels = []
        for window in range(num_windows):
            if window*step_size+window_size > data.shape[0]:
                break
            features.append(data[window*step_size:window*step_size+window_size, 1:-1])
            labels.append(data[window*step_size:window*step_size+window_size, -1])

        features = np.array(features).astype(np.float64)
        y_data = np.array(labels).astype(int)

        features = np.swapaxes(features, 1, 2)
        
        return features, y_data

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        if self.load_once:
            return self.features[idx], self.labels[idx]
        else:
            path = self.paths[idx]
            vector = self.load_data_car(path, columns_to_import=list("timestamp,latitude,longitude,speed,speed_x,speed_y,acceleration,acceleration_x,acceleration_y,angle,acc_diff,desc".split(",")))
            vector[:,-1], numero_clusters = self.translate_labels(vector[:,-1])
            vector = vector.astype(float)
            features, labels = self.window(vector, self.window_size, self.step_size)
            if self.normalization == "standard":
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True) + 1e-8
                features = (features - mean) / std
            return features, labels

    def load_data_car(self, path, columns_to_import=None, remove_nulled_columns=False, silent = True):
        if os.path.exists(path):
            # show the collums of the csv
            vector = pd.read_csv(path)
            if columns_to_import is None:
                columns_to_import = ["timestamp", "latitude", "longitude", "acceleration_x", "acceleration_y", "angle"]
            for col in columns_to_import:
                if col not in vector.columns:
                    vector[col] = 0
            existing_columns = [col for col in columns_to_import if col in vector.columns]
            # Forces the order of columns according to columns_to_import, keeping only the existing ones
            vector = vector.loc[:, existing_columns]
            if remove_nulled_columns:
                # Remove columns with all values as 0
                vector = vector.loc[:, (vector != 0).any(axis=0)]

            if not silent:
                print("Colunas finais:", list(vector.columns))
            if "timestamp" in vector.columns:
                vector["timestamp"] = vector["timestamp"] * 1000 + 1748858400000 # just ajust the timestamp to a more reasonable value
            vector = vector.values
        else:
            raise Exception(f'File not found: {path}')
        return vector
    
    def translate_labels(self, labels):
        """
        Translates string labels to integer labels. If the episode is not contiguous, it is broken in multiples contiguous.
        """
        current_label = None
        current_episode = -1
        episodes = []
        for i in range(len(labels)):
            if labels[i] != current_label:
                current_episode+=1
                current_label = labels[i]
            episodes.append(current_episode)
        episodes = np.array(episodes)
        return episodes, current_episode+1
    
class Pretrain_Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.video_lengths = []
        self.cumulative_lengths = [0]
        
        # Calculate lengths of each video
        for idx in range(len(dataset)):
            features, labels = dataset[idx]
            video_length = len(features)
            self.video_lengths.append(video_length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + video_length)
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        # Find which video this index belongs to
        video_idx = 0
        for i in range(len(self.cumulative_lengths) - 1):
            if idx >= self.cumulative_lengths[i] and idx < self.cumulative_lengths[i + 1]:
                video_idx = i
                break
        
        # Calculate the local index within that video
        local_idx = idx - self.cumulative_lengths[video_idx]
        
        # Get the video data
        features, labels = self.dataset[video_idx]
        
        # Return the specific frame
        return features[local_idx], labels[local_idx]
    
class BreakfastFramesDataset:
    def __init__(self, datasets_path, load_once = False):
        """
        Dataset class for Breakfast dataset with frames.
        
        Args:
            datasets_path (str): Path to the dataset directory containing IDs of subjects, ids os cams, activities and features. Example of expected structure: "Breakfast-frames/P03/cam01/P03_cereals/frame_000001.jpg"
        """

        self.path_ds = datasets_path

        # Find all directories containing .jpg files
        self.filenames = []
        jpg_folders = []
        for root, dirs, files in os.walk(self.path_ds):
            # print(files)
            if any(f.endswith('.jpg') for f in files):
                jpg_folders.append(root)
        jpg_folders.sort(key = lambda x: os.path.basename(x))

        # For each folder with jpgs, create entry with features path and corresponding labels path
        self.data_paths = []
        self.labels_paths = []
        for jpg_folder in jpg_folders:
            folder_name = os.path.basename(jpg_folder)
            parent_folder = os.path.dirname(jpg_folder)
            label_file = os.path.join(parent_folder, f"{folder_name}.avi.labels")
            if os.path.exists(label_file):
                self.data_paths.append(jpg_folder)
                self.labels_paths.append(label_file)

        self.gt_paths = [f + ".avi.labels" for f in self.data_paths]

        self.data, self.labels, self.n_clusters = dict(), dict(), dict()
        self.length = len(self.data_paths)
        assert self.length > 0, "No data found in the specified dataset path."
        assert len(self.data_paths) == len(self.labels_paths), "Mismatch between data paths and label paths lengths."
        if load_once:
            for idx in tqdm(range(self.length), desc="Loading Breakfast Frames Dataset"):
                self.load_video(idx)



    def get_n_clusters(self, idx):
        if idx not in self.n_clusters:
            self.load_item(idx)
        return self.n_clusters[idx]

    def __len__(self):
        return self.length
    
    def load_video(self, idx):
        frames = []
        for frame_file in sorted(os.listdir(self.data_paths[idx])):
            if frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.data_paths[idx], frame_file)
                frame = plt.imread(frame_path)
                frames.append(frame)

        # Load the GT_labels, map them to the corresponding ID    
        # Parse label file to get activity segments
        label_path = self.labels_paths[idx]
        with open(label_path, 'r') as f:
            label_lines = f.readlines()

        # Create mapping from activity names to IDs
        activity_to_id = {}
        current_id = 0
        for line in label_lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                activity_name = parts[1]
                if activity_name not in activity_to_id:
                    activity_to_id[activity_name] = current_id
                    current_id += 1

        # Create label vector with same length as number of frames
        num_frames = len(frames)
        frame_labels = np.full(num_frames, -1, dtype=np.int32)
        last_activity_id = None
        last_end = 0
        for line in label_lines:
            parts = line.strip().split()
            assert len(parts) >= 2
            frame_range = parts[0].split('-')
            start_frame = int(frame_range[0])
            assert start_frame == last_end , "Frame ranges in label file are not continuous."
            end_frame = int(frame_range[1])
            last_end = end_frame
            activity_name = parts[1]
            activity_id = activity_to_id[activity_name]
            
            # Fill frames in this range with the activity ID
            frame_labels[start_frame:end_frame] = activity_id
            last_activity_id = activity_id
        
        # Fill any remaining frames with the last activity ID
        if last_end < num_frames:
            frame_labels[last_end:num_frames] = last_activity_id
        

        # Extract activity name and video name from path for n_clusters lookup
        video_name = os.path.basename(self.data_paths[idx])
        activity_name = video_name.split('_', 1)[1] if '_' in video_name else video_name

        n_clusters = int(len(activity_to_id))

        self.data[idx] = np.array(frames)
        self.labels[idx] = frame_labels
        self.n_clusters[idx] = n_clusters

    def __getitem__(self, idx):
        if idx not in self.data:
            self.load_video(idx)
        return self.data[idx], self.labels[idx]

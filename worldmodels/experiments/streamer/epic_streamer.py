import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from repro.streamer.streamer.models.model import StreamerModel, StreamerModelArguments
from torch.utils.data import DataLoader
from tqdm import tqdm
from experiments.datasets import Epic_dataset
import numpy as np
from src.utilities import metricas, groups_to_episodes, streamer_metrics, common_hierarchical_cluster_representation
import lightning as L

class CNNEncoder(nn.Module):
    r"""
    A 4-layer CNN Encoder model used encode an image into a feature vector

    :param int feature_dim: the output feature dimension
    """

    def __init__(self, feature_dim):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 1024 , kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1024 * 8 * 8, feature_dim)

    
    def forward(self, x):
        r"""
        The forward propagation function that takes input image and returns output vector

        :param torch.Tensor x: tensor of shape [1, 3, H, W]
        :returns:
            * (*torch.Tensor*): feature vector of shape [1, feature_dim]
            * (*None*): Used for compatibility to return attention in other models
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 1024 * 8 * 8)
        x = self.fc1(x)
        return x, None


class CNNDecoder(nn.Module):
    r"""
    A 4-layer CNN Decoder model used decode a feature vector back to an image

    :param int feature_dim: the input feature dimension
    """

    def __init__(self, feature_dim):
        super(CNNDecoder, self).__init__()

        self.fc1 = nn.Linear(feature_dim, 1024 * 8 * 8)
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        r"""
        The forward propagation function that takes a feature vector and returns an image

        :param torch.Tensor x: tensor of shape [1, feature_dim]
        :returns:
            (*torch.Tensor*): image tensor of shape [1, 3, H, W]
        """

        x = self.fc1(x)
        x = x.view(-1, 1024, 8, 8)
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2)
        x = self.conv4(x)

        return x


def epic_streamer(seed):
    # Execute the experiment with the Streamer technique on the EPIC-KITCHENS dataset
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Para multi-GPU
    # L.seed_everything(seed)
    np.random.seed(seed)

    device = "cuda"
    model = StreamerModel(name = "epic_streamer", feature_dim=1024, encoder=CNNEncoder, decoder=CNNDecoder, n_heads=8, device = device, init_layers = 10, max_layers=10, snippet_size=1)
    print("Loading epic dataset")
    loader = DataLoader(dataset= Epic_dataset(data_path='datasets/epic_kitchens/frames/', load_once=False, resize = (128,128)), batch_size=1, num_workers= 8, pin_memory=True)
    # loader = DataLoader(dataset= Epic_dataset(data_path='datasets/epic_kitchens/frames/',ground_truths_path="datasets/epic_kitchens/groundTruth/", load_once=False, resize = (128,128)), batch_size=1, num_workers= 8, pin_memory=True)
    torch.manual_seed(0)
    
    print("Loaded epic dataset with length: ", len(loader))

    results = {"accuracy": 0, "f1_macro": 0, "iou": 0, "iou_streamer": 0, "mof_streamer": 0}
    valid_samples = 0
    for video_idx, video_gt in enumerate(loader):
        if video_gt[0] is None or len(video_gt[0]) == 0 or len(video_gt[0][0]) == 0:
            continue
        video, gt = video_gt[0][0], video_gt[1][0]
        valid_samples += 1
        print("Processing video index: ", video_idx, "with length: ", len(video))
        for frame in tqdm(video):
            model(frame.to(device).unsqueeze(0))
        print("Processed video of path: ", loader.dataset.paths[video_idx])
        ps, hierarchy = model.get_episode_labels(len(video))
        predicted_sequences = np.array(ps)
        common_rep = common_hierarchical_cluster_representation(predicted_sequences, save_json=f"experiments/results/epic_streamer/{seed}/{video_idx}/common_rep.json")
        gt = gt.squeeze().cpu().numpy()
        gt = groups_to_episodes(gt)
        best_f1, best_i = 0, -1
        for i, predicted_sequence_level in enumerate(predicted_sequences):
            level_result = metricas(gt, predicted_sequence_level, path_to_save_fig=f"experiments/results/epic_streamer/{seed}/{video_idx}/layer_{i}/", save_data_as_npy=True, correct_labels = True)
            if level_result["f1_macro"] > best_f1:
                best_f1 = level_result["f1_macro"]
                best_i = i
        result = metricas(gt, predicted_sequences[best_i], path_to_save_fig=f"experiments/results/epic_streamer/{seed}/{video_idx}/", save_data_as_npy=True, correct_labels = True)
        results["accuracy"] += result["accuracy"]
        results["f1_macro"] += result["f1_macro"]
        results["iou"] += result["iou"]
        strem_metr = streamer_metrics(len(video), hierarchy, loader.dataset.get_json(video_idx), gt_idx=0)
        print("Streamer Metrics (IoU, MoF): ", strem_metr)
        results["iou_streamer"] += strem_metr[0]
        results["mof_streamer"] += strem_metr[1]
        model.reset_model()

    print("all samples processed, calculating final results")

    results["accuracy"] /= valid_samples
    results["f1_macro"] /= valid_samples
    results["iou"] /= valid_samples
    results["iou_streamer"] /= valid_samples
    results["mof_streamer"] /= valid_samples
    return results


if __name__ == "__main__":
    print(epic_streamer(seed=0))
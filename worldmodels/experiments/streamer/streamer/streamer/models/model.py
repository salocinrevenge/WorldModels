import torch, os
import torch.nn as nn
import torch.nn.functional as F
try:
    from streamer.models.layer import StreamerLayer, StreamerLayerArguments
    from streamer.arguments.base_arguments import parser
    from streamer.utils.logging import setup_output
    import streamer.optimizers as optimizers
except:
    from repro.streamer.streamer.models.layer import StreamerLayer, StreamerLayerArguments
    from repro.streamer.streamer.arguments.base_arguments import parser
    from repro.streamer.streamer.utils.logging import setup_output
    import repro.streamer.streamer.optimizers as optimizers

from dataclasses import dataclass, asdict

@dataclass
class StreamerModelArguments():

    log_base_every: int = 1000
    r"""Tensorboard log of the base layer every 'log_base_every'"""

    main: bool = True
    r"""Main process/gpu or not"""

    max_layers: int = 3
    r"""The maximum number of layers to stack"""

    feature_dim: int = 1024
    r"""Feature dimension of the model embeddings"""

    evolve_every: int = 50000
    r"""Create/stack a new layer every 'evolve_every' """

    buffer_size: int = 20
    r"""Maximum input buffer size to be used"""

    force_fixed_buffer: bool = False
    r"""Force the buffer to be fixed (not replacing inputs) by triggering a boundary when buffer is full"""

    loss_threshold: float = 0.25
    r"""Loss threshold value. Not used in average demarcation mode"""

    lr: float = 1e-4
    r"""Learning rate to be used in all modules"""

    init_layers: int = 1
    r"""How many layers to initialize before training"""

    init_ckpt: str = ''
    r"""the path of the pretrained weights, if any"""

    ckpt_dir: str = ''
    r"""the path to save weights"""

    snippet_size: float = 0.5
    r"""Snippet size of input video (seconds/image). Typically 0.5 seconds per image"""

    demarcation_mode: str = 'average'
    r"""Demarcation mode used to detect boundaries"""

    distance_mode: str = 'distance'
    r"""Distance mode for loss calculation"""

    force_base_dist: bool = True
    r"""Force the lowest layer to use MSE instead of Cosine Similarity"""

    window_size: int = 50
    r"""Window size for average demarcation mode"""

    modifier_type: str = 'multiply'
    r"""Modifier type to apply to average demarcation mode ['multiply', 'add']"""

    modifier: float = 1.0
    r"""Modifier to apply to avrrage demarcation mode"""

    @staticmethod
    def from_args(args):
        return StreamerModelArguments(
                init_ckpt=args.init_ckpt,
                init_layers=args.init_layers,
                ckpt_dir=args.ckpt_dir,
                snippet_size=args.snippet_size,
                log_base_every=args.log_base_every,
                main=args.main,
                max_layers=args.max_layers,
                feature_dim=args.feature_dim,
                evolve_every=args.evolve_every,
                buffer_size=args.buffer_size,
                force_fixed_buffer=args.force_fixed_buffer,
                loss_threshold=args.loss_threshold,
                lr=args.lr,
                demarcation_mode=args.demarcation_mode,
                distance_mode=args.distance_mode,
                force_base_dist=args.force_base_dist,
                window_size=args.window_size,
                modifier_type=args.modifier_type,
                modifier=args.modifier,
        )


    @staticmethod
    def from_ckpt_args(args):
        return StreamerModelArguments(
                init_ckpt=args.init_ckpt,
                init_layers=args.init_layers,
                ckpt_dir=args.ckpt_dir,
                snippet_size=args.snippet_size,
                log_base_every=args.log_base_every,
                main=args.main,
                max_layers=args.max_layers,
                feature_dim=args.feature_dim,
                evolve_every=args.evolve_every,
                buffer_size=args.buffer_size,
                force_fixed_buffer=args.force_fixed_buffer,
                loss_threshold=args.loss_threshold,
                lr=args.lr,
                demarcation_mode=args.demarcation_mode,
                distance_mode=args.distance_mode,
                force_base_dist=args.force_base_dist,
                window_size=args.window_size,
                modifier_type=args.modifier_type,
                modifier=args.modifier,
                fp_up=args.fp_up,
                fp_down=args.fp_down
        )



class StreamerModel(nn.Module):
    r"""
    The implementation of the STREAMER model for training. This class initializes the first layer(s) of streamer, saves/loads weights, etc.

    :param StreamerModelArguments args: The arguments passed to Streamer Model
    :param Logger logger: The tensorboard logger class
    :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
    :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
    """

    def __init__(self, name:str, feature_dim:int, logger=None, encoder=None, decoder=None, n_heads = 8, device = "cuda", init_layers = 10, max_layers = 10, snippet_size = 1, save_every = 200000, json_dir: str = None):
        super(StreamerModel, self).__init__()
        self.logger = logger
        self.streamer = None
        self.models_saved = 0
        self.base_counter = 0
        self.init_layers = init_layers
        self.max_layers = max_layers
        self.snippet_size = snippet_size # time in seconds between time stamps
        self.name = name # name of the experiment
        self.save_every = save_every
        self.device = device
        self.feature_dim = feature_dim # size of the feature vector extracted by the encoder or input size when no encoder is used
        if json_dir is None:
            json_dir = f"experiments/out/{name}/"
        self.json_dir = json_dir
        os.makedirs(self.json_dir, exist_ok=True)
        self.get_args()
        self.args.buffer_size = 20
        self.__initialize(self.args.init_ckpt, self.args.init_layers, encoder=encoder, decoder=decoder, n_heads=n_heads, device = device)

        self.orig_args.module = 'repro.streamer.streamer.optimizers.streamer_optimizer' # to import on func bellow
        self.optimizer = optimizers.getOptimizer(self.orig_args, self)

    def get_args(self):
        args = parser().parse_args()
        args.init_layers = self.init_layers
        args.max_layers = self.max_layers
        args.dataset="."
        args.snippet_size = self.snippet_size
        args.dbg = True
        args.main = True
        args.name = self.name
        args.logger = self.logger
        args.save_every = self.save_every
        args = setup_output(args)
        args.device = self.device
        args.feature_dim = self.feature_dim  # tamanho da janela
        args.json_dir = self.json_dir
        self.args = StreamerModelArguments.from_args(args)
        self.orig_args = args


    def init_layer(self, count=1, encoder=None, decoder=None, n_heads = 8):
        r"""
        Initializes the Streamer layer(s) and passes the enocder/decoder to the first layer.
        
        :param int count: How many layers to create 
        :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
        :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
        :param int: The number os atentions heads
        """

        # Arguments
        self.args.reps_fn = self.getReps
        streamerLayerArgs = StreamerLayerArguments.from_model_args(self.args)

        self.streamer = StreamerLayer(
                args = self.args,
                layer_num = 0, 
                init_count = count,
                encoder=encoder,
                decoder=decoder,
                logger=self.logger,
                n_heads = n_heads,
                )


    def __initialize(self, ckpt='', count=1, encoder=None, decoder=None, n_heads = 8, device = "cpu"):
        r"""
        Initializes the Streamer layer(s) with checkpoint if available.
        
        :param str ckpt: pretrained weights location
        :param int count: How many layers to create 
        :param torch.nn.Module encoder: The encoder model (e.g., :py:class:`~streamer.models.networks.CNNEncoder`)
        :param torch.nn.Module decoder: The decoder model (e.g., :py:class:`~streamer.models.networks.CNNDecoder`)
        """

        if ckpt=='':
            self.init_layer(count=count, encoder=encoder, decoder=decoder, n_heads=n_heads)

        else:
            ckpt= torch.load(ckpt, map_location='cpu')
            self.init_layer(count=ckpt['num_layers'], encoder=encoder, decoder=decoder, n_heads=n_heads)
            self.streamer.load_state_dict(ckpt['weights'])
        self.streamer.to(device)
            

    def forward(self, x):
        r"""
        Forward propagation function that calls the :py:meth:`~streamer.models.layer.StreamerLayer.forward` function of the first :py:class:`~streamer.models.layer.StreamerLayer`.
        
        :param torch.Tensor x: the input image [1, 3, H, W]
        """
        # log base signal
        if self.logger != None: 
            self.logger.model(self.streamer.context, 
                              x, 
                              self.streamer.attn_img, 
                              self.streamer.attns)

        # main forward
        self.streamer(x, base_counter=self.base_counter)
        self.base_counter += 1

    def train_step(self, x):
        r"""
        Trining step function instead of forward for training purposes.
        """

        self.forward(x)
        
        self.optimizer.step()


    def getReps(self, layer_num):
        r"""
        Aggregates the representations from all the layers.

        :param int layer_num: the index of the calling layer

        :returns:
            (*torch.Tensor*): concatenated representations from all layers [L, feature_dim]

        """
        curr_layer_num = 0
        ll = self.streamer
        reps = [ll.representation]

        while ll.above != None and ll.above.representation != None:
            curr_layer_num += 1
            ll = ll.above
            reps.append(ll.representation)

        return torch.cat(reps, dim=0)

    def extract_rep(self):
        r"""
        Extract representation function used for logging json hierarchy. 
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.extract_rep` on the :py:class:`~streamer.models.layer.StreamerLayer` class

        :returns:
            (*dict*): hierarchy represented as boundaries of every layer

        """
        if self.streamer == None:
            return {}

        hierarchy = dict(boundaries = {})
        self.streamer.extract_rep(hierarchy)

        # close open boundaries
        for layer_name, layer_bounds in hierarchy['boundaries'].items():
            if layer_name == 0: continue
            if layer_bounds[-1] != hierarchy['boundaries'][0][-1]:
                layer_bounds.append(hierarchy['boundaries'][0][-1])

        return hierarchy

    def reset_model(self):
        r"""
        Resets the whole streamer model for a new video.
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.reset_layer` on the :py:class:`~streamer.models.layer.StreamerLayer` class

        """
        if self.streamer == None:
            return 

        self.streamer.reset_layer()
        self.base_counter = 0

    def get_num_layers(self):
        r"""
        Get the total number of layers.
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.get_num_layers` on the :py:class:`~streamer.models.layer.StreamerLayer` class
        """
        if self.streamer == None:
            return 0

        return self.streamer.get_num_layers(0)

    def optimize_model(self):
        r"""
        Optimizes the whole streamer model (gradient step). 
        Calls recursive function :py:meth:`~streamer.models.layer.StreamerLayer.optimize_layer` on the :py:class:`~streamer.models.layer.StreamerLayer` class
        """
        if self.streamer != None:
            self.streamer.optimize_layer()
        

    def save_model(self):
        r"""
        Saves the model weights to :py:class:`~StreamerModelArguments.ckpt_dir`
        """

        if self.streamer == None or self.args.main == False:
            return 

        ckpt = dict(
                model_args = asdict(self.args),
                model_modality = self.modality,
                model_snippet_size = self.args.snippet_size,
                num_layers = self.get_num_layers(), 
                weights = self.streamer.state_dict())

        torch.save(ckpt, os.path.join(self.args.ckpt_dir, f'model_{str(self.models_saved).zfill(3)}.pth'))
        self.models_saved += 1

    def get_episode_labels(self, size_time_series):
        r"""
        Converts the hierarchical boundaries extracted by :py:meth:`~extract_rep` into episode labels for each time step in the dataset.

        :param Dataset dataset: The dataset used to get the number of time steps

        :returns:
            (*list*): A list of predicted episode labels for each time step in the dataset

        """
    # Convert hierarchical sequence into episode labels
        predicted_sequences = []
        episode_boundaries = self.extract_rep()['boundaries']  # Assuming hierar['boundaries'] contains multiple layers

        for layer in episode_boundaries:  # Iterate through each layer of boundaries
            boundaries = episode_boundaries[layer]
            predicted_sequence = []
            for i in range(size_time_series):  # For each time step
            # Find which episode this time step belongs to
                for j in range(len(boundaries) - 1):
                    if boundaries[j] <= i < boundaries[j + 1]:
                        predicted_sequence.append(j)
                        break
            predicted_sequences.append(predicted_sequence)  # Store the predicted sequence for this layer
            # Append final episode for any remaining time steps
            while len(predicted_sequence) < size_time_series:
                predicted_sequence.append(len(episode_boundaries[layer]) - 1) 
        return predicted_sequences, episode_boundaries

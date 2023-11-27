from .utils.preprocessing_tools import get_index, check_and_summarize
from .utils.processing_tools import remove_inf_node, remove_nan_node
import sys
import numpy as np
from numpy import random
import torch
from torch_geometric.loader import DataLoader as pyg_dataloader
from copy import deepcopy
import os.path as osp
import json

class BaseIndexingSystem(object):
    """
    An indexing system based on metadata defined on multiple files.
    
    Example of allowed metadata:
        {"1" :  {"file1":"62", "file2":"13", "file3":"24"},
         "2" :  {"file2":"2", "file5":"33", "file6":"245"},
         "14":  {"file1":"21", "file2":"23", "file3":"34"}}
    Here, we have three classes of data distributed in six files.
    
    For many puposes, we may want to collect a sample flom these 
    ambigious databases without distruction and recunstruction.
    
    This system let us to map a given index into a data inside 
    the database. Also, it allows us to do sampling over database.
    """
    
    def __init__(self, collection, seed, **sampling_strategy):
        self._seed                        = seed
        self._metadata                    = metadata    # Metadata
        self._metadata_summary            = dict()      # Metadata Summary
        self._sample_metadata             = dict()      # Sample Metadata
        self._sample_metadata_information = dict()      # Sample Metadata Information (= Summary)
        self._filenames                   = None        # File Names Container
        self._num_sampled_data            = None        # Number of sampled data we can access through this IndexingSystem
        self._num_total_data              = None        # Number of total data we can access through this IndexingSystem
        self._summarize()
        self._smaple_from_collection(collection, **sampling_strategy)
        
    def _summarize(self):
        """ 
        Summarizing the metadata:
            1. List of files
            2. Dictionary for pu - #pu pair; i.e. DICT[PU] = #PU
        """
        files = set()
        for PU in self._MD:
            PU_metadata = self._metadata[PU]
            current_PU = 0
            for file_name in PU_metadata:
                files.add(file_name)
                current_PU += int(PU_metadata[file_name])
               
            self._metadata_summary[PU] = current_PU
        self._filenames = list(files)
                        
    def _data_collection_preprocessing(self, data_col:dict):
        """
        Data collection preprocessing:
            1. filtering mising pu's
            2. Avoiding overflow
        """
        rejected = dict()
        for PU in data_col:
            try:
                wanted_PU    = data_col[PU]
                available_PU = self._metadata_summary[PU]
            except KeyError:
                print(f"PU = {PU} is not available. Excluding PU = {int(PU):03d} ...", file=sys.stderr)
                rejected[PU] = data_col[PU]
                # Excluding ...
                continue
            if available_PU < wanted_PU:
                w = f"PU = {int(PU):03d} overflows... Setting PU = {int(PU):03d} to {available_PU} instead of {wanted_PU}"
                print(w, file=sys.stderr)
                wanted_PU = available_PU
                
            self._sample_metadata[PU] = wanted_PU
        self._num_sampled_data = np.sum(list(self._sample_metadata.values()))
        return rejected
    
    def _smaple_from_collection(self, dc:dict, **sampling_kwargs):
        """
        Random sampling of the whole data from sample metadata
        """
        random.seed(self._seed)
        for PU in self._sample_metadata:
            PU = str(int(PU))
            self._sample_metadata_information[PU] = random.choice(
                  range(int(self._metadata_summary[PU])),
                  size=int(self._sample_metadata[PU]), 
                  **sampling_kwargs
                  )        
        
    def _map_index(self, ind):
        """
        `ind` is a "global" index i.e. defind over all "sampled" data.
        Trivially, each "global" index is reffered to a "local" index.
        It is mapped into a local index of a given PU associated with `ind`.
        """
        PU, ind = get_index(self._sample_metadata, ind)
        real_index = self._sample_meatdata_information[PU][int(ind)]
        return PU, real_index
    
    def _get_item(self, ind):
        """
        This method do the final job of index processing system.
        It returns all information in order to load the data.
        """
        PU, PU_index = self._map_index(ind) 
        PU_MD = self._metadata[PU]
        file_name, infile_index = get_index(PU_MD, PU_index)
        return file_name, PU, infile_index
    
    @property
    def num_available_data(self):
        """
        Returns the number of sampled data
        """
        return self._num_sampled_data
   
    @property
    def num_all_data(self):
        """
        Returns the number of allowed data
        """
        return self._num_total_data

    @property
    def files(self):
        """
        Returns the file names
        """
        return self._filenames
    
    @property
    def keys(self):
        """
        Returns available PU classes
        """
        return self._sample_metadata.keys()


class BaseDataReader(object):
    def remove_nan_nodes(self, nodes, edges, edge_attrs):
        has_nan = np.isnan(nodes).sum() > 0
        if has_nan:
            return has_nan, remove_nan_node(node_features=nodes, edge_index=edges, edge_attributes=edge_attrs)
        return has_nan, None
    
    def remove_inf_nodes(self, nodes, edges, edge_attrs):
        has_inf = np.isinf(nodes).sum() > 0
        if has_inf:
            return has_inf, remove_inf_node(node_features=nodes, edge_index=edges, edge_attributes=edge_attrs)
        return has_inf, None

    def __call__(self, *args, **kwargs):
        return get(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        raise NotImplementedError("This is a abstract class")
        

class BaseDataset(object):
    def __init__(self, root, in_dir, sample_metadata, reader, seed=42,
                 sampling_strategy={"replace":False, "p":None},
                 transform=None, pre_transform=None, pre_filter=None, log=False):
        
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        self._in_dir       = in_dir
        self.root          = root
        self.transform     = transform
        self.pre_transform = pre_transform
        self.pre_filter    = pre_filter
        self.log           = log
        self._indices      = None
        self._files        = dict()
        self._data_reader  = BaseDataReader()
        
        if self.log:
            print('Processing...', file=sys.stderr)
        self.process(sample_metadata, **sampling_strategy)
        if self.log:
            print('Done!', file=sys.stderr)
                
    def process(self, sample_metadata, **kwargs):
        """Process if you need"""
        with open(osp.join(self.root, 'metadata.json'), "r") as jsonfile:
            metadata_dict = json.load(jsonfile)
        
        self._indexing_system = BaseIndexingSystem(metadata_dict, sample_metadata, self.seed, **kwargs)
        
    
    def len(self):
        """Returns length of dataset i.e. numberr of graphs in dataset"""
        return int(self._indexing_system.num_available_data)
    
    def indices(self):
        """Returns indices array in increasing order"""
        return range(self.len()) if self._indices is None else self._indices

    def get(self, idx):
        """
        Returns the data associated with `idx` under our inherited `IndexingSystem`
        """
        raise NotImplementedError("This is an abstract class")
    
    @property
    def num_node_features(self) -> int:
        r"""Returns the number of features per node in the dataset."""
        data = self[0] 
        
        if hasattr(data, 'num_node_features'):
            return data.num_node_features
        
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_node_features'")
    
    @property
    def num_edge_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        if hasattr(data, 'num_edge_features'):
            return data.num_edge_features
        
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'num_edge_features'")
    
    @property
    def num_features(self) -> int:
        r"""Returns the number of features per edge in the dataset."""
        data = self[0]
        if hasattr(data, 'features'):
            return len(data.features)
        
        raise AttributeError(f"'{data.__class__.__name__}' object has no "
                             f"attribute 'features'")
    

class BaseDataloader(object):
    def __init__(self, dataset, seed=42, cache_dir=None,
            test_precentage=10, validation_precentage=10,
            batch_size=64, num_workers=4, shuffle=False, pin_memory=False, **kwargs
            ) -> None:
        self._seed = seed
        self._root = cache_dir
        self._batch_size  = dataloader_args['batch_size']
        self._dataset     = dataset
        self.process(len(dataset), 10, 10, batch_size=64, num_workers=4, shuffle=False, pin_memory=False, **kwargs)
        # Contexmanager vars:
        self._open = False
        self._context_loader   = None
        self._context_dir      = None
        self._context_metadata = None
        self._context_device   = None
        
    def process(self, dataset_length, test_percentage, validation_percentage, **dataloader_args):
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        self._validation_len = (dataset_length * test_percentage) // 100 
        self._test_len       = (dataset_length * validation_percentage) // 100
        self._train_len      = dataset_length - self._validation_len - self._test_len
        self._len = {
            'test': self.test_len, 
            'validation': self.validation_len,
            'train': self.train_len
        }

        train_subset, valid_subset, test_subset = torch.utils.data.random_split(
                self._dataset, lengths=[self._train_len, self._validation_len, self._test_len]
            )
        # Generators
        self._train_gen       = pyg_dataloader(train_subset, **dataloader_args)
        self._validation_gen  = pyg_dataloader(valid_subset, **dataloader_args)
        self._test_gen        = pyg_dataloader(test_subset,  **dataloader_args)
        self._loader_metadata = dict()

    def __enter__(self, subset, device='cuda:0'):
        assert subset in ['test', 'train', 'validation'], f"{subset} is unknown. Use one of 'test', 'train', 'validation'."
        self._open = True
        self._context_loader   = subset
        self._context_device   = device
        self._context_dir      = osp.join(self._root, subset)
        self._context_metadata = self._loader_metadata[subset]
        yield iter(self)
    
    @property
    def test_len(self):
        return self._test_len

    @property
    def train_len(self):
        return self._train_len

    @property
    def validation_len(self):
        return self._validation_len

    def total(self):
        if self._open:
            return self._len[self._context_loader]
        return len(self._dataset)

    def __len__(self):
        if self._open:
            return len(self._loader_metadata[self.self._context_loader])
        raise AssertionError

    def __exit__(self):
        self._open = False

    def __iter__(self):
        if not self._open:
            raise RuntimeError("Iteration only availabele when you open the dataloader")
        raise NotImplementedError


class BaseTrainer(object):
    def __init__(self, root, dataloader, seed=42):
        self._prog    = None
        self._device  = None
        self._model   = None
        self._root    = osp.join(root, 'trainer')
        self._seed    = seed
        self._loader  = dataloader
        self._optimizer    = None
        self._loss_func    = None
        self._lr_scheduler = None

        self._models_dir   = os.path.join(self._root, 'models')
        ind = 0
        while osp.isdir(self._models_dir):
            ind += 1
            self._models_dir = os.path.join(self._root, f'models({ind})')
        os.mkdir(self._models_directory)
    
    def __enter__(self, model, device='cuda:0', disable_progress_bar=False):
        self._prog   = disable_progress_bar
        self._moder  = model
        self._device = device
        self._prog   = disable_progress_bar
        return self

    @property
    def dataloader(self):
        return self._loader
    
    @property
    def dataloader(self):
        return self._loader

    @property
    def model(self):
        return self._model
    
    @property
    def device(self):
        return self._device
    
    def check_data(self, data, b_ind):
        return check_data(data, b_ind)
    
    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
        
    def train(self, max_epochs, optimizer, optimizer_args,
                                loss_fn, loss_fn_args=dict(),
                                lr_scheduler=None, lr_scheduler_args=dict(),
                                use_benchmark=True, metrics=[], select_topk=5, **kwargs):
            
        print("Training device {}...\n".format(self._device))
        self._model        = self._model.to(self._device)
        self._loss_func    = loss_fn(**loss_fn_args)
        self._optimizer    = optimizer(self._model.parameters(), **optimizer_args)
        self._lr_scheduler = lr_scheduler(self._optimizer, **lr_scheduler_args) if lr_scheduler is not None else None
        train_loss_arr = np.zeros((max_epochs, 1+len(metrics)))
        valid_loss_arr = np.zeros((max_epochs, 1+len(metrics)))
        is_failed = False
        torch.backends.cudnn.benchmark = use_benchmark
        summary = namedtuple('TrainingSummary', ['failing_status', 'res'])
        for epoch in trange(max_epochs, desc=f'Training the {self._name}...', unit='Epoch', ncols=950, disable=self._prog):
            torch.cuda.empty_cache()
            is_failed, res = self.train_one_epoch(epoch, grad_clipping_number)
            if is_failed:
                print(res[0], file=sys.stderr)
                return summary(is_failed, res)
            
            is_failed, res = self.evaluate("train", metrics)
            if is_failed:
                print(res[0], file=sys.stderr)
                return summary(is_failed, res)
            
            train_loss_arr[epoch] = res
            
            is_failed, res = self.evaluate('validation', metrics)
            if is_failed:
                print(res[0], file=sys.stderr)
                return summary(is_failed, res)
            
            valid_loss_arr[epoch] = res
            
            
            path_to_model = os.path.join(self._models_directory, f"epoch-{epoch + 1:0{len(str(max_epochs))}d}.pt")
            torch.save(self._model.state_dict(), path_to_model)
                        
            print(f'Training Set Loss: {train_loss_arr[epoch]:.4f}, Validation Set Loss: {valid_loss_arr[epoch]:.4f}\n')
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

        self._model = self._model.to('cpu')

        if save_models:
            kbest_model_ind = np.argmin(valid_loss_arr, )
            kbest_model_dir = os.path.join(self._models_directory, f"epoch-{best_model_ind + 1:0{len(str(max_epochs))}d}.pt")
            kbest_model_new_dir = os.path.join(self._models_directory, f"epoch-{best_model_ind + 1:0{len(str(max_epochs))}d}(best).pt")
            os.rename(best_model_dir, best_model_new_dir)
            self._model.load_state_dict(torch.load(best_model_dir))

        fig = go.Figure(
                data = [
                    go.Scatter(x=list(range(1, max_epochs + 1)), y=train_loss_arr, name='Training Set'),
                    go.Scatter(x=list(range(1, max_epochs + 1)), y=valid_loss_arr, name='Validation Set'),
                ]
        )

        fig.update_layout(title="Training Summary")
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title=str(self._loss_func)[:-2])
        
        plotly.io.write_html(fig,  os.path.join(self._root, 'Training Summary.html'))
        plotly.io.write_json(fig,  os.path.join(self._root, 'Training Summary.json'))
        
        summary = namedtuple('TrainingSummary', 
            ['failing_status', 'training_set_loss', 'validation_set_loss', 'plot', 'models']
        )        
        
        return summary(is_failed, train_loss_arr, valid_loss_arr, fig, self._bestk_models(valid_loss_arr, select_topk))
    
    
class MultibatchPUGNNTrainer(PUGNNTrainer):
    def __init__(self, num_batch=5, *args, **kwargs):
        super(MultibatchPUGNNTrainer, self).__init__(*args, **kwargs)
        self._n = num_batch
        
    def train_one_epoch(self, epoch=-1, clip=None):
        self._model.train()  
        b_ind = -1
        data_list = []
        last_ind = self._train_len - 1
        
        for data in tqdm(self._train_gen, desc=f"Epoch {epoch+1:03d}", unit="Batch", disable=self._prog):
            b_ind += 1
            is_last = b_ind == last_ind
            
            clean, err = self.check_data(data, b_ind)
            
            if not clean:
                print(err, file=sys.stderr)
                continue
                
            if (b_ind % self._n == self._n - 1) or is_last:
                data_list.append(data)
                out    = self._model(data_list)    # Perform a single forward pass.
                target = torch.cat([d.y for d in data_list])
                loss   = self._loss_func(out, target.unsqueeze(1))                           # Compute the loss.

                if np.isnan(loss.item()):
                    w = "nan loss detected. Perhaps there is a divergance. Stopping the training..."
                    return 1, (w, data)
                
                loss.backward()  # Derive gradients.
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), clip)
                self._optimizer.step()       # Update parameters based on gradients.
                self._optimizer.zero_grad()  # Clear gradients.
                
                del data_list
                data_list = []
            
            else:
                data_list.append(data)
            
        return 0, None
    
        
class BaseAnalyzer(object):
    def __init__(self, root, loader, seed=42):
        self._seed   = seed
        self._loader = loader
        self._metric = None
        self._models = None
        self._device = None
        self._model  = None
        self._yhat   = None
        self._y      = None
        self._range  = None
        self._output_dir = osp.join(root, 'analyzer')

        ind = 0
        while osp.isdir(self._main_dir):
            ind += 1
            self._output_dir = osp.join(root, f'analyzer({ind})')
        osp.mkdir(self._output_dir)

        print("processing...")
        self._process()
    
    def __enter__(self, models, metric, device='cuda:0'):
        self._metric = metric
        self._models = models
        self._device = device
        return self


    @property
    def yhat(self):
        return self._yhat
    
    @property
    def y(self):
        return self._y

    def _loss(self, model, loader):
        length   = len(loader)
        loss_arr = np.zeros(length)
        total = loader.all_data()
        failed = False
        # Iterate in batches over the training/test/validation dataset.
        for data in loader:
            out = model(data)  
            if out.cpu().detach().isnan().sum() > 0:
                w = f"nan loss detected during evaluation. Perhaps there is a problem..."
                failed = True
                return failed, (w, data)

            loss_arr[b_ind, ind] += metric(out, data.y).cpu().item() * len(data) / total
            
        return failed, loss_arr.sum(0)

    def process(self, num_models):
        with torch.no_grad(), open(self._loader, 'test') as test_loader:
            loss_values = []
            for model in self._models:
                model = model.to(self._device)
                model.eval()
                failed, loss = self._loss(test_loader, model)
                if failed:
                    raise ValueError
                loss_values.append(loss)

            self._model = self._models[np.argmin(loss_values)]
            self._yhat = torch.tensor([], dtype=float)
            self._y    = torch.tensor([], dtype=float)

            for data in tqdm(test_loader, desc='Evaluating...', unit='Batch', ncols=1000):
                out  = self._model(data)
                
                self._yhat = torch.concat([self._yhat, out.cpu().detach()])
                self._y = torch.concat([self._y, data.y.cpu().detach()])

            self._yhat = self._yhat.squeeze().detach().numpy()
            self._y    = self._y.detach().numpy()
            
            self._range = np.arange(self._y.min(), self._y.max() + 2)
    
    def histogram(self):
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=self._y, name='PU'))
        fig_hist.add_trace(go.Histogram(x=self._yhat, name='Estimated PU'))
        fig_hist.update_xaxes(title="PU")
        fig_hist.update_yaxes(title="Count")
        fig_hist.update_layout(title="Histogram of PU and estimated PU")
        plotly.io.write_json(fig_hist,  os.path.join(self._output_dir, 'histogram.json'))
        plotly.io.write_html(fig_hist,  os.path.join(self._output_dir, 'histogram.html'))
        return fig_hist
    
    def kde_plot(self):
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Histogram2dContour(
                x = self._y,
                y = self._yhat,
                colorscale = 'plasma',
                reversescale = False,
                xaxis = 'x',
                yaxis = 'y'
            ))

        fig_kde.add_trace(go.Histogram(
                y = self._yhat,
                xaxis = 'x2',
                marker = dict(
                    color = 'rgba(0,0,100,1)'
                )
            ))
        fig_kde.add_trace(go.Histogram(
                x = self._y,
                yaxis = 'y2',
                marker = dict(
                    color = 'rgba(0,0,100,1)'
                )
            ))

        fig_kde.update_layout(
            autosize = False,
            xaxis = dict(
                zeroline = False,
                domain = [0,0.7],
            ),
            yaxis = dict(
                zeroline = False,
                domain = [0,0.7],
            ),
            xaxis2 = dict(
                zeroline = False,
                domain = [0.7,1],
            ),
            yaxis2 = dict(
                zeroline = False,
                domain = [0.7,1],
            ),
            height = 800,
            width = 800,
            bargap = 0,
            hovermode = 'closest',
            showlegend = False
        )
        
        fig_kde.update_xaxes(title="PU")
        fig_kde.update_yaxes(title="Estimated PU")
        fig_kde.update_layout(title="KDE Plot of PU and Estimated PU")
        plotly.io.write_html(fig_kde,   os.path.join(self._output_dir, 'kdeplot.html'))
        plotly.io.write_json(fig_kde,   os.path.join(self._output_dir, 'kdeplot.json'))

    def heatmap(self):
        fig_hm = self.heatmap()
        fig_hm = go.Figure(go.Histogram2d(x=self._y, y=self._yhat))
        fig_hm.update_xaxes(title="PU")
        fig_hm.update_yaxes(title="Estimated PU")
        fig_hm.update_layout(title="2D histogrma of PU and Estimated PU")
        plotly.io.write_html(fig_hm,    os.path.join(self._output_dir, 'heatmap.html'))
        plotly.io.write_json(fig_hm,    os.path.join(self._output_dir, 'heatmap.json'))

    def distribution_plots(self):
        s = namedtuple("DistPlots", ["heatmap", "histogram", 'kdeplot'])
        return s(self.heatmap(), self.histogram(), self.kde_plot())
    
    def residual_plot(self):
        residuals = self._yhat - self._y
        fig = px.scatter(x=self._yhat, y=residuals)
        fig.update_yaxes(title='Residuals')
        fig.update_xaxes(title='Estimated PU')
        fig.update_layout(title="Residual Plot")
        plotly.io.write_html(fig,  os.path.join(self._output_dir, 'residual-plot.html'))
        plotly.io.write_json(fig,  os.path.join(self._output_dir, 'residual-plot.json'))
        return fig
    
    def compare(self, **outputs):
        RSS = (self._y - self._yhat)**2
        TSS = (self._y - self._y.mean())**2
        r_squared_array = np.zeros(len(outputs) + 1)
        r_squared_array[0] = 1 - RSS.sum() / TSS.sum()
        fig = go.Figure()
        for i, output_name in enumerate(outputs):
            y, yhat = outputs[output_name]
            fig.add_trace(go.Scatter(x=y, y=yhat, name=output_name, mode='markers'))
            RSS_p = (yhat - self._yhat)**2
            r_squared_array[i + 1] = 1 - RSS.sum() / RSS_p.sum()
        
        fig.add_trace(go.Scatter(x=self._y, y=self._yhat, name='The Model', mode='markers'))
        fig.add_trace(go.Scatter(x=self._range, y=self._range, name='Actual Trend'))
        fig.update_xaxes(title='No. PU')
        fig.update_layout(title='Comparing')
        
        plotly.io.write_html(fig,  os.path.join(self._output_dir, 'compare-models-plot.html'))
        plotly.io.write_json(fig,  os.path.join(self._output_dir, 'compare-models-plot.json'))
        
        s = namedtuple('Comparing', ['plot', 'R2'])
        return s(fig, r_squared_array)
    
    

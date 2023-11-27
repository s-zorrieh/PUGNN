from .base import BaseDataReader, BaseDataset, BaseDataloader, BaseTrainer, BaseAnalyzer
from .utils.processing_tools import get_data, to_data
from .utils.preprocessing_tools import check_and_summarize
import numpy as np
import datetime 
import sys
import os.path as osp
import os
import pytz
import os
import shutil



class HomoDataReader(BaseDataReader):
    def __call__(self, path_to_file, group, index, log=False):
        failed, res = get_data(path_to_file, group, index)
        if failed and log:
            print(res, file=sys.stderr)
            raise FileNotFoundError
        
        node_features = np.array(res["node_features"], dtype=np.float32)
        edge_attrs = np.array(res["edge_attributes"], dtype=np.float32)
        edge_index = np.array(res["edge_indecies"], dtype=np.int64)
        features   = np.array(res["graph_features"], dtype=np.float32)
        labels     = np.array(res["graph_labels"], dtype=np.float32)
                
        has_nan, res1 = self.remove_nan_nodes(node_features, edge_index, edge_attrs)
        has_inf, res2 = self.remove_inf_nodes(node_features, edge_index, edge_attrs)
        if has_nan:
                node_features, edge_attrs, edge_index = res1
                if log:
                    print("'{}' : 'PU={}' : 'E{}' : nan".format(path_to_file, group, index), file=sys.stderr)
        if has_inf:
                node_features, edge_attrs, edge_index = res2
                if log:
                    print("'{}' : 'PU={}' : 'E{}' : inf".format(path_to_file, group, index), file=sys.stderr)
        
        return to_data(node_features, edge_index, edge_attrs, features, labels)
        

class HeteroDataReader(BaseDataReader):
    def __call__(self, path_to_file, group, index, log=False):
        ...


class HomoDataset(BaseDataset):
    def get(self, idx):
        filename, PU, infile_index = self._indexing_system._get_item(self, idx)
        path_to_file = osp.join(self._in_dir, filename + ".h5")
        data = self._data_reader(path_to_file, f"PU{PU}", f"E{infile_index}")


class DataLoader(BaseDataloader):
    def process(self, dataset_length, dataset_args, dataloader_args):
        super().process(dataset_length, dataset_args, dataloader_args)
        self._context_metadata = dict(
            zip(['test', 'train', 'validation'], [self._test_gen, self._train_gen, self._validation_gen])
            )
    
    def __iter__(self):
        if not self._open:
            raise RuntimeError("Iteration only availabele when you open the dataloader")
        yield iter(self._context_metadata[self._context_loader])


class BoostedDataLoader(BaseDataloader):
    def process(self, dataset_length, dataset_args, dataloader_args):
        super().process(dataset_length, dataset_args, dataloader_args)
        print('Initializing...', file=sys.stderr)
        for name in ['test', 'train', 'validation']:
            osp.mkdir(self._root, name)
        for subset, name in zip([self._test_gen, self._train_gen, self._validation_gen], ['test', 'train', 'validation']):
            zeros  = len(str(len(subset)))
            branch = osp.join(self._root, name)
            for ind, data in enumerate(subset):
                path_to_data = osp.join(branch, f'batch_{ind:0{zeros}d}.pt')
                torch.save(data, path_to_data)
            self._loader_metadata[name] = [f'batch_{ind:0{zeros}d}.pt'for ind in range(len(self._test_gen))]
        print('done.', file=sys.stderr)

    def __iter__(self):
        if not self._open:
            raise RuntimeError("Iteration only availabele when you open the dataloader")

        self._iters      = 0
        self._last_iter  = len(self._loader_metadata[self._context_loader])
        return self
    
    def __next__(self):
        if self._iters == self._last_iter:
            self._reset_contextmanager
            raise StopIteration
        batch = self._context_metadata[self._iters]
        path_to_batch = osp.join(self._context_dir, batch)
        self._iters += 1
        data = torch.load(path_to_batch)
        return data.to(self._context_device)
    

class Trainer(object):    
    def train_one_epoch(self, epoch=-1):
        self._model.train()
        with open(self.dataloader, 'train', self.device) as train_loader:

            for data in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", unit="Batch", disable=self._prog):
                out  = self._model(data)   # Perform a single forward pass.
                loss = self._loss_func(out, data.y.unsqueeze(1))
                if np.isnan(loss.item()):
                    w = "nan loss detected. Perhaps there is a divergance. Stopping the training..."
                    return 1, (w, data)
                loss.backward()              # Derive gradients.
                self._optimizer.step()       # Update parameters based on gradients.
                self._optimizer.zero_grad()  # Clear gradients.
            return 0, None

    def evaluate(self, subset, metrics=[]):
        self._model.eval()
        b_ind = 0 

        failed = False
        
        with torch.no_grad(), open(self.dataloader, subset, self.device) as loader:
            length   = len(loader)
            loss_arr = np.zeros((length, 1+len(metrics)))
            total = loader.all_data()
            
            # Iterate in batches over the training/test/validation dataset.
            for data in loader:
                data = data.to(self._device)
                out = self._model(data)  
                if out.cpu().detach().isnan().sum() > 0:
                    w = f"nan loss detected during evaluation of 'training set'. Perhaps there is a problem..."
                    failed = True
                    return failed, (w, data)
                for ind, metric in enumerate([self._loss_func, *metrics]):
                    loss_arr[b_ind, ind] += metric(out, data.y.unsqueeze(1)).cpu().item() * len(data) / total
                b_ind += 1
        return failed, loss_arr.sum(0)

class Analyzer(BaseAnalyzer):
    def _poisson_dist_extraction(self, av, f=0.6):
        prv = poisson(av)
        n  = len(self._y) / len(self._range)
        N  = int(f*n / prv.pmf(av))
        freq       = np.histogram(prv.rvs(N), self._range)[0]
        pu_freq    = dict(zip(self._range, freq))
        pu_counter = dict(zip(self._range, freq))
        pu_inds    = []

        for i, data in enumerate(tqdm(self._y,  desc=f'PU = {av} Indexing...', unit='Graph')):
            y = int(data)
            if pu_counter[y] > 0:
                pu_inds.extend([i])
                pu_counter[y] -= 1
        
        if sum(list(pu_counter.values())) > 1:
            w = f"Change factor variable `fraction` (current value is `{f}`) based on the following:"
            print(w, file=sys.stderr)
            for pu in pu_counter:
                if pu_counter[pu] > 0:
                    w = f'\tPoisson distribution({av}) for PU = {pu} is violated by {pu_counter[pu]} count less than expected'
                    print(w, file=sys.stderr)
        return pu_inds
    
    def LLEstimation(self, av_pu, f=0.6):
        inds = self._poisson_dist_extraction(av_pu, f)
        xhat = self._yhat[inds]
        x    = self._y[inds]
        
        return max_log_likelihood(xhat, llikelihood_pois, av_pu), x.mean()
    
    def rangeLLE(self, starting_pu, ending_pu, fraction=0.6):
        av_PUs = np.array([*range(starting_pu, ending_pu + 1)])

        mle_arr       = np.zeros(len(av_PUs))
        act_arr       = np.zeros_like(mle_arr)
        left_err_arr  = np.zeros_like(mle_arr)
        right_err_arr = np.zeros_like(mle_arr)

        for ind, pu in enumerate(tqdm(av_PUs, desc='Range-Based LLE',  unit='PU', ncols=950)):
            temp, act = self.LLEstimation(pu, fraction)
            le, mle, re = temp
            
            mle_arr[ind]       = mle
            act_arr[ind]       = act
            left_err_arr[ind]  = le
            right_err_arr[ind] = re
        
        fig = go.Figure(data=go.Scatter(
            x=av_PUs,
            y=mle_arr,
            error_y=dict(
                type='data',
                symmetric=False,
                array=right_err_arr,
                arrayminus=left_err_arr,
                visible=True), name='Estimated <PU>', mode='markers')
                )
        
        fig.add_trace(go.Scatter(x=av_PUs, y=av_PUs, name='Expected'))
        fig.update_xaxes(title='$<PU>$')
        fig.update_layout(title='Model Reliabilty for Poisson Distribution')

        s = namedtuple('LogLikelihood_PU_Estimation',
                       ['plot', 'true_pu', 'estimated_pu', 'lower_bond_error', 'upper_bond_error'])
        
        plotly.io.write_html(fig,  os.path.join(self._output_dir, 'model-reliabilty-poisson-dist.html'))
        plotly.io.write_json(fig,  os.path.join(self._output_dir, 'model-reliabilty-poisson-dist.json'))

        return s(fig, act_arr, mle_arr, left_err_arr, right_err_arr)

class SW(object):
    def __init__(self, root, input_dir, name) -> None:
        self._time = str(datetime.datetime.now(pytz.timezone('Asia/Tehran'))).split('.')[0].replace(' ', '@')
        self._root = root
        self._name = name
        self._in_dir       = input_dir
        self._main_dir     = osp.join(root, f'{name}-pugnnsw')
        self._cache_dir    = osp.join(self._main_dir, f'cache-{self._time}')    
        self._output_dir   = osp.join(self._main_dir, f'out-{self._time}')
        self._metadata_dir = osp.join(self._main_dir, 'metadata')

        if not os.path.isdir(self._main_dir):
            os.mkdir(self._main_dir)

        if not os.path.isdir(self._metadata_dir):
            os.mkdir(self._metadata_dir)

        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)

        check_and_summarize(self._in_dir, self._metadata_dir)
    
    @property
    def root(self):
        return self._root

    @property
    def name(self):
        return self._name

    @property
    def main_directory(self):
        return self._main_dir

    @property 
    def dataset(self):
        return self._dataset

    @property
    def loader(self):
        return self._loader
    
    def set_dataset(self, Dataset, sample_metadata, data_reader, **kwargs) -> None:
        self._dataset = Dataset(self._metadata_dir, self._in_dir, sample_metadata, self._seed, data_reader, **kwargs)

    def set_loader(self, DataLoaderClass, **kwargs):
        self._loader = DataLoaderClasss(self.dataset, self._seed, self._cache_dir, **kwargs)
    
    @property
    def trainer_scope(self):
        return Trainer(self._output_dir, self.loader, self._seed)

    @property
    def analyzer_scope(self):
        return Analyzer(self._output_dir, self.loader, self._seed)

    def __del__(self):
        shutil.rmtree(self._cache_dir)


    
    

    
	


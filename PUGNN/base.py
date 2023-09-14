import sys
import numpy as np
from numpy import random
from utils.preprocessing_tools import get_index, check_and_summarize, get_data, to_data
from utils.processing_tools import remove_inf_node, remove_nan_node
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
        raise NotImplementedError("This is a abstract class")
        


class BaseDataset(object):
    def __init__(self, root, sample_metadata, seed,
                 sampling_strategy={"replace":False, "p":None}, reader=BaseDataReader,
                 transform=None, pre_transform=None, pre_filter=None, log=False):
        
        if isinstance(root, str):
            root = osp.expanduser(osp.normpath(root))
        
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
        with open(osp.join(root, 'metadata'), "r") as jsonfile:
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
    

class Dataset(BaseDataset):
    def get(self, idx):
        filename, PU, infile_index = self._indexing_system._get_item(self, idx)
        
        path_to_file = osp.join(self.root, filename + ".h5")
        
        data = self._data_reader(path_to_file, f"PU{PU}", f"E{infile_index}")
        
        
        

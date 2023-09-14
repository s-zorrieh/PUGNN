from base import BaseDataReader, BaseDataset
from utils.processing_tools import get_data, to_data
import numpy as np
import datetime 
import sys
import os.path as osp
import pytz
import os
import shutil

from contextlib import contextmanager


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


class Dataset(BaseDataset):
    def get(self, idx):
        filename, PU, infile_index = self._indexing_system._get_item(self, idx)
        
        path_to_file = osp.join(self.root, filename + ".h5")
        
        data = self._data_reader(path_to_file, f"PU{PU}", f"E{infile_index}")

class SW(object):
    def __init__(self, root, name, boost=True, clear_cache=False) -> None:
        self._time = str(datetime.datetime.now(pytz.timezone('Asia/Tehran'))).split('.')[0].replace(' ', '@')
        self._root = root
        self._name = name

        self._main_dir     = osp.join(root, f'{name}-pugnnsw')
        self._metadata_dir = osp.join(self._main_dir, 'metadata')
        self._output_dir   = osp.join(self._main_dir, 'out')
        self._cache_dir    = osp.join(self._main_dir, f'cache-{self._time}')

        if not os.path.isdir(self._main_dir):
            os.mkdir(self._main_dir)
            
        if not os.path.isdir(self._metadata_dir):
            os.mkdir(self._metadata_dir)

        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)
        
        if boost:
             os.mkdir
             
	
    def __del__(self):
        if self.destruct:
            shutil.rmtree(self._cache_dir)
    

            
            
    
        

class BoostedDataLoader:
    def __init__(self, root, clear=True) -> None:
        self._cache_dir = osp.join(root, 'bdl-cache')
        if osp.isdir(self._cache_dir) and clear:
            shutil.rmtree(self._cache_dir)
            os.mkdir(self._cache_dir)
            self.fill_buffer()
        else:
        	print("The cache is used from privious run", file=sys.stderr)
                
    def fill_buffer(self):
        file_lists = os.listdir(self._cache_dir)
        self._buffer = ...
    
	@contextmanager
    def test(self):
         pass
        

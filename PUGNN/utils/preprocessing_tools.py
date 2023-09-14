import h5py
import json
import os
import os.path as osp
import time
import sys
from torch_sparse import SparseTensor   





################################################
def is_ext(string:str, ext0='json'):
    
    res = string.split('.')
    if (len(res) != 2) or (string.find('.') < 0):
        return False, string
 
    name, ext = res
    return (ext == ext0), name

def check_and_summarize(directory, to='./', rewrite=True, log=True):
    nfiles = 0
    json_files = []
    summary = dict()
    
    with open(osp.join(to, 'metadata.log'), "w") as logfile:
        
        for file in os.listdir(directory):
            is_hdf5_file, file_name = is_ext(file, 'h5')

            if is_hdf5_file:
                try:
                    with h5py.File(directory + file) as f:
                        pass

                    with open(osp.join(directory, file_name + '.json'), "r") as j:
                        d = json.load(j)
                        nfiles += 1
                        for pu in d:
                            try:
                                summary[pu][file_name] = d[pu][file_name]
                            except KeyError:
                                summary[pu] = d[pu]

                except Exception as e:
                    print(file, ':', e, file=logfile)
                    if log:
                        print(file, ':', e, file=sys.stderr)

    # summary
    if rewrite:
        with open(os.path.join(to, 'summary.json'), "w") as outfile:
                outfile.write(json.dumps(summary, indent=4))
    return nfiles



def get_index(counter_dict, ind):
        """
        Assuming `counter_dict` has a shape as `self._MDSummary` i.e. {cls_i: `freq of cls_i`}.
        In this case, indexing is over "All" classes not a specific one. 
        Knowing the frequency of each class, we iterate over `counter_dict` to find "local" index.
        In summary, we want to find the `class` and local index corresponding to a given `ind`.
        """
        index = ind
        total = 0
        for key in counter_dict:
            index -= counter_dict[key]
            total += counter_dict[key]
            if index < 0 :
                index += counter_dict[key]
                return key, index
        raise IndexError(f"The index {ind} is out of range [0, {total}).")


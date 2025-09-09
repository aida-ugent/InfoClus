import cProfile
import infoclus
import os
from config import PROJECT_ROOT
from src.caching import from_cache

emb_name = 'tsne'
linkage = 'single'
modify_hierarchical = True
data_name = 'Mouse1_Batch1_WT_50K'
file_path = os.path.join(PROJECT_ROOT, 'data', data_name, 'cache',
                         emb_name + '_' + linkage + '_' + 'modify_' + str(modify_hierarchical))

if os.path.exists(file_path):
    print('loading ' + file_path)
    infoclus_obj = from_cache(file_path)
    print('done')
else:
    infoclus_obj = infoclus.InfoClus(dataset_name=data_name, emb_name=emb_name, linkage=linkage,
                                     modify_hierarchical=modify_hierarchical)

def main():
    infoclus_obj.optimise(run_id=4)

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.print_stats(sort='time')

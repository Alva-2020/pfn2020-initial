
import csv
import os
from pathlib import Path

# DEPRECATED. Load directly into dataset. see dataset.py

def list_dataset(dataset_name:str) -> int:
    r"""Creates the following files in data/dataset/list/, given a dataset name.
    Expects files and directories arranged as follows:

    root
        - scripts
            - (this script lives here)
        - data
            - Directory called <datasetname> that you passed to this script
                - raw
                    - Directories called <domain_name>
                        - Directories called <class_name>
                            - Image files
                - list (created if not present)
                    - meta (created if not present)

    map_domains.tsv:        Maps domain names to [0-D] integer indices
    map_class.tsv:          Maps class names to [0-C] integer indices
    counts.tsv:             For each domain, n instances of each class. This is a matrix.
    <domain_name>_full.tsv:      For each domain, map of file path to class indices
    """

    CWD = os.getcwd()
    if os.path.basename(CWD) == 'scripts':
        print("Please set working directory to /scripts. Exiting.")
        return 1
    
    DATA_DIR = 'data/'
    DATASET_DIR = os.path.join(DATA_DIR, dataset_name)
    RAW_DIR = os.path.join(DATASET_DIR, 'raw')
    LIST_DIR = os.path.join(DATASET_DIR, 'list')
    META_DIR = os.path.join(LIST_DIR, 'meta')

    if not os.path.exists(RAW_DIR):
        print("Raw directory not found. Exiting.")
        return 1
    if not os.path.exists(LIST_DIR):
        Path(LIST_DIR).mkdir(parents=False, exist_ok=False)
    if not os.path.exists(META_DIR):
        Path(META_DIR).mkdir(parents=False, exist_ok=False)


    domain2idx = {}
    class2idx = {}

    # Create map_domain.tsv
    with open(os.path.join(META_DIR, 'domains.tsv'), 'w+') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')

        f.seek(0)
        for d, domain_name in enumerate(os.listdir(RAW_DIR)):
            writer.writerow([d, domain_name])
            domain2idx[domain_name] = d
        f.truncate()
        f.close()
    
    # Create map_class.tsv
    # Assumes all classes accounted for in each domain in a single dataset,
    # so just read from the first domain dir.
    domain_dir = os.path.join(RAW_DIR, os.listdir(RAW_DIR)[0])
    with open(os.path.join(META_DIR, 'classes.tsv'), 'w+') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')

        f.seek(0)
        for c, class_name in enumerate(os.listdir(domain_dir)):
            writer.writerow([c, class_name])
            class2idx[class_name] = c
        f.truncate()
        f.close()

    # Create counts.tsv
    with open(os.path.join(META_DIR, 'counts.tsv'), 'w+') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')

        f.seek(0)
        for domain_name, d in domain2idx.items():
            strlist = []
            strlist.append(domain_name)
            for class_name, c in class2idx.items():
                path = os.path.join(RAW_DIR, domain_name, class_name)
                count = len(os.listdir(path))
                strlist.append(count)
            writer.writerow(strlist)
        f.truncate()
        f.close()

    # Create full list for each domain
    for domain_name, d in domain2idx.items():
        with open(os.path.join(LIST_DIR, 'full_{}.tsv'.format(domain_name)), 'w+') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            f.seek(0)
            for class_name, c in class2idx.items():
                path = os.path.join(RAW_DIR, domain_name, class_name)
                for filename in os.listdir(path):
                    pathstr = os.path.join(path, filename)
                    writer.writerow([pathstr, c])
            f.truncate()
            f.close()



if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentError

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', action='store')
    args = parser.parse_args()

    if args.dataset in {'officehome'}:
        list_dataset(args.dataset)
    else:
        print("Dataset not recognized. Exiting.")

    
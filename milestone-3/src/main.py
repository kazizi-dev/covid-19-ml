import os, warnings
from ada_boost import get_ada_boost_results

if __name__ == '__main__':
    print('-------------------------------------------------------------')
    warnings.filterwarnings("ignore")

    # setup path for direcotry files
    os.chdir(os.getcwd())

    # train models
    path = '../data/cases_less.csv'
    get_ada_boost_results(path)
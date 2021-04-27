import os, warnings
from ada_boost import start_testing_ada_boost
from random_forest import start_testing_random_forest

#### for Windows Sbusystem Linux enter 1, for mac entetr 2
SYSTEM_INPUT = 1

if __name__ == '__main__':
    print('-------------------------------------------------------------')
    warnings.filterwarnings("ignore")

    # setup path for direcotry files
    os.chdir(os.getcwd())

    print('....testing AdaBoost')
    start_testing_ada_boost(SYSTEM_INPUT)
    
    print('....testing RandomForst')
    start_testing_random_forest(SYSTEM_INPUT)

    print("Status: SUCCESS")

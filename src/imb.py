import os

import pandas as pd

input_directory = 'dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))
number_of_bugs = (nodes_df['NUMBER-OF-BUGS'] > 0).sum()
number_of_not_bugs = (nodes_df['NUMBER-OF-BUGS'] <= 0).sum()

print(number_of_bugs)
print(number_of_not_bugs)

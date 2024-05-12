#DATASET LINK
https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb 


# IEEE-ML-PROJECT-I04
 !pip install opendatasets
import opendatasets as od

# Kaggle dataset link
dataset_link = "https://www.kaggle.com/datasets/safabouguezzi/german-traffic-sign-detection-benchmark-gtsdb"

# Download the dataset
od.download(dataset_link)
import os

# Define the directory where the dataset is downloaded
dataset_dir = "/content/german-traffic-sign-detection-benchmark-gtsdb"

# List the contents of the dataset directory
dataset_contents = os.listdir(dataset_dir)
print("Contents of the dataset directory:")
print(dataset_contents)
import os
import pandas as pd

# Define the directory where the dataset is downloaded
dataset_dir = "/content/german-traffic-sign-detection-benchmark-gtsdb"

# Load the dataset with Pandas
annotations_df = pd.read_csv(os.path.join(dataset_dir, 'gt.txt'), sep=';', header=None)

# Display first few rows of the DataFrame
print(annotations_df.head())





 
          
 
                    
                    
                    
                    
                    
                    


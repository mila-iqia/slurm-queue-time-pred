import csv
import os

# Path to the original DRAC format data files
path_to_data = os.path.join(os.path.dirname(__file__), '../../original_data')

# Path to the data files with the rows that are matching the date in the file name
path_to_data_on_submit_days = os.path.join(os.path.dirname(__file__), '../../data_on_submit_days')

    
# I have data files in the format of: jobs_cedar_2021-05-01.csv
# I need to compare the date in the file name (after the second underscore) 
# to the date in the column "submit_time" in the file and remove the rows that aren't
# matching the date in the file name.
def remove_rows_not_matching_date_in_file_name():
    # list files in all subdirectories of path_to_data
    data_files = [file_name for (_,_,file_names) in os.walk(path_to_data) for file_name in file_names if file_name.endswith(".csv")]
    print("files:", len(data_files))
    for data_file in data_files:
        # get the date from the file name
        date = data_file.split("_")[2].split(".")[0]
        # Get cluster name from file name, e.g. "cedar" from "jobs_cedar_2021-05-01.csv"
        cluster = data_file.split("_")[1]
        # open the file
        with open(f"{path_to_data}/{cluster}/{data_file}", "r") as f:
            reader = csv.reader(f)
            # get the header
            header = next(reader)
            # get the index of the column "submit_time"
            submit_time_index = header.index("submit_time")
            # get the rows that match the date in the file name
            rows = [row for row in reader if row[submit_time_index].startswith(date)]   
        # write the rows to a new file with same name in a folder with the cluster name in directory "data_on_submit_days"
        with open(f"{path_to_data_on_submit_days}/{cluster}/{data_file}", "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
            
        
if __name__ == "__main__":
    remove_rows_not_matching_date_in_file_name()
    
"""
python3 preprocessing.py
"""

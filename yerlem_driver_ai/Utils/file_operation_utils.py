from pandas import DataFrame, read_csv

class FileOperationUtils:
    def __init__(self):
        pass

    # DESCRIPTION: Load data from a CSV file
    def load_data(self, file_path):
        try:
            data = read_csv(file_path)
            return data
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return DataFrame()
    
    # DESCRIPTION: Save data to a CSV file
    def save_data(self, data, file_path):
        if data is not None:
            try:
                # DESCRIPTION: Load existing data from the file
                existing_data = self.load_data(file_path)
                # DESCRIPTION: If the file does not exist, create a new file
                if existing_data.empty:
                    data.to_csv(file_path, index=False)
                    print(f"Data saved: {file_path}. Index: {data.shape[0]}")
                else:
                    # DESCRIPTION: Append data to the existing file
                    data.to_csv(file_path, mode='a', header=False, index=False)
                    new_index = existing_data.shape[0] + data.shape[0]
                    print(f"Data saved: {file_path}. Index: {new_index}")
            except Exception as e:
                print(f"Error saving data: {e}")
        else:
            print("No data to save.")
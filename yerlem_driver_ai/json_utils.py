import json

class JsonUtils:
    def __init__(self):
        pass
    
    def load_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                # DESCRIPTION: Read the JSON file and load the data.
                return json.load(f)
        except FileNotFoundError:
            return []
        
    def get_next_id(self, data):
        if data:
            # DESCRIPTION: Get the maximum id from the data and increment it by 1. This allows for unique ids to be assigned to new data.
            return max(item.get('id', 0) for item in data) + 1
        else:
            return 1

    # DESCRIPTION: Save data to a JSON file.
    def save_data(self, data, file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    # DESCRIPTION: Save the captured data to a JSON file with the label and id.
    def save_captured_data(self, points, data, label, id, file_path):
        data.append({"id": id, "label": label, "landmarks": points})
        self.save_data(data, file_path)
        print(f"Captured data saved: {file_path}, {id}, {label}")
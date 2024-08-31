import json

def pretty_print_json(json_data):
    print(json.dumps(json_data, indent=4, sort_keys=True))

# Example usage
if __name__ == "__main__":
    sample_json = {
        "name": "Fantasy Sports Optimizer",
        "version": "1.0",
        "description": "Optimize your fantasy sports teams with AI."
    }
    pretty_print_json(sample_json)

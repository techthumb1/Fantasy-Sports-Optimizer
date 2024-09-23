import json

def pretty_print_json(json_data):
    """
    Converts JSON data into a pretty-printed JSON string.

    Args:
        json_data (dict): The JSON data to be formatted.

    Returns:
        str: A pretty-printed JSON string.
    """
    return json.dumps(json_data, indent=4, sort_keys=True)

# Example usage
if __name__ == "__main__":
    sample_json = {
        "name": "Fantasy Sports Optimizer",
        "version": "1.0",
        "description": "Optimize your fantasy sports teams with AI."
    }
    print(pretty_print_json(sample_json))

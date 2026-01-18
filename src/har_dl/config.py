import yaml
import os
from har_dl.definitions import get_roots

project_root, package_root = get_roots()

def load_config(config_path: str = os.path.join(project_root, "config.yaml")) -> dict:
    """
    Loads a YAML configuration file and returns its content as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.
                           Defaults to 'config.yaml' in the project root.

    Returns:
        dict: A dictionary containing the configuration settings,
              with 'data' paths resolved to absolute paths.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

if __name__ == "__main__":
    print(f"Project Root: {project_root}")
    print(f"Package Root: {package_root}")

    config_data = load_config()

    if config_data:
        print("\nConfiguration loaded successfully:")
        print(yaml.dump(config_data, indent=2))
    else:
        print("Failed to load configuration.")
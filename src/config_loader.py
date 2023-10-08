from yaml import safe_load, SafeLoader

# singular function to read out the contents of the config file
def config_loader(config_location: str):
    with open(config_location, 'r') as yaml_file:
        config_data = safe_load(yaml_file) 

    return config_data


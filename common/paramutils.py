def get_param(config, param_path, default_value):
    path_list = param_path.split(".")
    config_ = config
    for item in path_list:
        if item in config_:
            config_ = config_[item]
        else:
            return default_value
    return config_

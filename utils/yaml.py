import yaml


def save_cfg(path, train_data, test_data, cfg_shape_net, cfg_parameter_net, mixed_policy):

    with open(path + 'cfg.txt', 'w') as f:
        f.write('{')
        f.write(f'path : {path} \n')
        f.write(f'train samples: {train_data.shape[0]}\n')
        f.write(f'test samples: {test_data.shape[0]}\n')
        f.write(f'cfg_shape_net : {cfg_shape_net} \n')
        f.write(f'cfg_parameter_net : {cfg_parameter_net} \n')
        f.write(f'mixed_policy : {mixed_policy} \n')
        f.write('}')
        return True


def save_cfg_yaml(path, train_data, test_data, cfg_shape_net, cfg_parameter_net, mixed_policy, train_cfg=None):
    d = {'path': path,
         'train_samples': train_data.shape[0],
         'test_samples': test_data.shape[0],
         'cfg_shape_net': cfg_shape_net,
         'cfg_parameter_net': cfg_parameter_net,
         'mixed_policy': mixed_policy,
         'training_cfg': train_cfg}
    with open(path + 'cfg.yaml', 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)
    return True


def read_cfg_yaml(file):
    with open(file, 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.Loader)
    path = opt['path']
    train_samples = opt['train_samples']
    test_samples = opt['test_samples']
    cfg_shape_net = opt['cfg_shape_net']
    cfg_parameter_net = opt['cfg_parameter_net']
    mixed_policy = opt['mixed_policy']
    train_cfg = opt['training_cfg']
    return path, train_samples, test_samples, cfg_shape_net, cfg_parameter_net, mixed_policy, train_cfg


def yaml2dict(file):
    with open(file, 'r') as stream:
        opt = yaml.load(stream, Loader=yaml.Loader)
    return opt


def dict2yaml(file, dct):
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(dct, outfile, default_flow_style=False, allow_unicode=True)
    return True
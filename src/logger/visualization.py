from .wandb import WanDBWriter


def get_visualizer(config, logger, type):
    if type == 'wandb':
        return WanDBWriter(config, logger)

    raise NotImplementedError

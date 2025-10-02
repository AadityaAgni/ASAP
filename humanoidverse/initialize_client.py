import os
from socket import socket
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

import logging
from loguru import logger
import zmq




from utils.config_utils import *  # noqa: E402, F403


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    # import ipdb; ipdb.set_trace()
    simulator_type = config.simulator['_target_'].split('.')[-1]
    # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacSim':
        from omni.isaac.lab.app import AppLauncher
        import argparse
        parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
        AppLauncher.add_app_launcher_args(parser)
        
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = config.env.config.env_spacing # config.env_spacing
        args_cli.output_dir = config.output_dir
        args_cli.headless = config.headless
        
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app  
        
        # import ipdb; ipdb.set_trace()
    if simulator_type == 'IsaacGym':
        import isaacgym  # noqa: F401


    # have to import torch after isaacgym
    import torch  # noqa: E402
    from utils.common import seeding
    import wandb
    from humanoidverse.envs.base_task.base_task import BaseTask  # noqa: E402
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    from humanoidverse.utils.logging import HydraLoggerBridge
        
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())
    
    if hasattr(config, 'device'):
        if config.device is not None:
            device = config.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    pre_process_config(config)
    
    ### Initializing Simulator
    print(config.keys())
    print(config.env.keys())
    
    SimulatorClass = get_class(config.simulator._target_)
    print(f"Initializing simulator: {SimulatorClass}")
    print(config.keys())
    sim = SimulatorClass(config=config, device=device)
        
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{config.zmq_port}")
    
    while True:
        print(f"Waiting for request... at port {config.zmq_port}")
        message = socket.recv_pyobj()
        attr = message.get('attr')
        if hasattr(sim, attr):
            attr_value = getattr(sim, attr)
            if callable(attr_value):

                args = message.get('args', [])
                kwargs = message.get('kwargs', {})
                result = attr_value(*args, **kwargs)
                socket.send_pyobj(result)

                print(f"Received request for method: {attr} with args: {args} and kwargs: {kwargs}. Sent response: {result}")
            else:
                socket.send_pyobj(getattr(sim, attr))

                print(f"Received request for attribute: {attr}. Sent response: {getattr(sim, attr)}")
        elif attr == 'exit':
            print("Exiting simulation client.")
            socket.send_pyobj("Exiting")
            if simulator_type == 'IsaacSim':
                simulation_app.close()
            break
        else:
            socket.send_pyobj(None)

if __name__ == "__main__":
    main()

from humanoidverse.simulator.base_simulator.base_simulator import BaseSimulator
import torch
import numpy as np
import zmq

class CrossSim(BaseSimulator):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.sim_device = device
        self.headless = False
        
        
        self.simulator_clients, self.env_counts = self.create_simulator_clients()

        
    
    def create_simulator_clients(self):
        clients = {}
        env_counts = {}
        for simulator_name in self.config.simulator_names:
            sim_config = self.config.sim_configs[simulator_name]
            
            port = sim_config.get('port', None)
            assert port, f"Port must be specified for simulator {simulator_name} in crosssim config."
            # Initialize ZMQ client for each simulator
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://localhost:{port}")

            clients[simulator_name] = socket
            
            env_counts[simulator_name] = 0
        return clients, env_counts

    # ----- Configuration Setup Methods -----
    def set_headless(self, headless):
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'set_headless', 'args': [headless]})
            message = client.recv_pyobj()
        self.headless = headless
    
    def setup(self):
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'setup'})
            message = client.recv_pyobj()
        
    # ----- Terrain Setup Methods -----
    def setup_terrain(self, mesh_type):
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'setup_terrain', 'args': [mesh_type]})
            message = client.recv_pyobj()
    
    # ----- Robot Asset Setup Methods -----
    def load_assets(self):
        for client in self.simulator_clients.values()   :
            client.send_pyobj({'attr': 'load_assets', 'args': [self.config.robot]})
            message = client.recv_pyobj()

    # ----- Environment Creation Methods -----

    def create_envs(self, num_envs, env_origins, base_init_state):
        self.env_counts = self.get_env_counts(num_envs)

        curr_idx = 0
        for simulator_name, count in self.env_counts.items():
            for _ in range(count):
                params = [
                    count,
                    env_origins[curr_idx:curr_idx + count],
                    base_init_state
                ]
                curr_idx += count
                self.simulator_clients[simulator_name].send_pyobj({'attr': 'create_env', 'args': params})
                message = self.simulator_clients[simulator_name].recv_pyobj()
        assert curr_idx == num_envs, "Total environments created do not match num_envs."
        
     # ----- Property Retrieval Methods -----

    def get_dof_limits_properties(self):
        """
        Retrieves the DOF (degrees of freedom) limits and properties.
        
        Returns:
            Tuple of tensors representing position limits, velocity limits, and torque limits for each DOF.
        """
        client = self.simulator_clients.values().__iter__().__next__() # Get any client to query DOF properties
        client.send_pyobj({'attr': 'get_dof_limits_properties'})
        message = client.recv_pyobj()
        return message['pos_limits'], message['vel_limits'], message['torque_limits']
    
    def find_rigid_body_indice(self, body_name):
        """
        Finds the index of a specified rigid body.

        Args:
            body_name (str): Name of the rigid body to locate.

        Returns:
            int: Index of the rigid body.
        """
        index = None
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'find_rigid_body_indice', 'args': [body_name]})
            message = client.recv_pyobj()
            
            # This assert is basically a TODO from Aaditya to log when we have inconsistent indices, since this could be difficult to handle
            assert index is None or index == message, "Inconsistent body indices across simulators."
            
            index = message
        return index
    
    # ----- Simulation Preparation and Refresh Methods -----

    def prepare_sim(self):
        """
        Prepares the simulation environment and refreshes any relevant tensors.
        """
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'prepare_sim'})
            message = client.recv_pyobj()
            
    
    def refresh_sim_tensors(self):
        """
        Refreshes the state tensors in the simulation to ensure they are up-to-date.
        """
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'refresh_sim_tensors'})
            message = client.recv_pyobj()

    # ----- Control Application Methods -----

    def apply_torques_at_dof(self, torques):
        """
        Applies the specified torques to the robot's degrees of freedom (DOF).

        Args:
            torques (tensor): Tensor containing torques to apply.
        """
        # Torques are a (num_envs, num_dofs) tensor
        start_idx = 0
        for simulator_name, count in self.env_counts.items():
            end_idx = start_idx + count
            client = self.simulator_clients[simulator_name]
            client.send_pyobj({'attr': 'apply_torques_at_dof', 'args': [torques[start_idx:end_idx]]})
            message = client.recv_pyobj()
            start_idx = end_idx
        assert start_idx == torques.shape[0], "Total torques applied do not match number of environments."
    
    def set_actor_root_state_tensor(self, set_env_ids, root_states):
        """
        Sets the root state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            root_states (tensor): New root states to apply.
        """
        # Set_env_ids is a (some_val) tensor
        # root_states is a (num_envs, some_other_vals) tensor
        # Complexity here is that these ids need to be split across_simulators
        separated_ids = self.separate_env_ids(set_env_ids)
        for simulator_name, env_ids in separated_ids.items():
            if env_ids.shape[0] == 0:
                continue
            client = self.simulator_clients[simulator_name]
            client.send_pyobj({'attr': 'set_actor_root_state_tensor', 'args': [env_ids, root_states[env_ids]]})
            message = client.recv_pyobj()
            
    
    def set_dof_state_tensor(self, set_env_ids, dof_states):
        """
        Sets the DOF state tensor for specified actors within environments.

        Args:
            set_env_ids (tensor): Tensor of environment IDs where states will be set.
            dof_states (tensor): New DOF states to apply.
        """
        # Set_env_ids is a (some_val) tensor
        # dof_states is a (num_envs, some_other_vals, another_val) tensor
        # Complexity here is that these ids need to be split across_simulators
        separated_ids = self.separate_env_ids(set_env_ids)
        for simulator_name, env_ids in separated_ids.items():
            if env_ids.shape[0] == 0:
                continue
            client = self.simulator_clients[simulator_name]
            client.send_pyobj({'attr': 'set_dof_state_tensor', 'args': [env_ids, dof_states[env_ids]]})
            message = client.recv_pyobj()

    def simulate_at_each_physics_step(self):
        """
        Advances the simulation by a single physics step.
        """
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'simulate_at_each_physics_step'})
            message = client.recv_pyobj()

    # ----- Viewer Setup and Rendering Methods -----

    def setup_viewer(self):
        """
        Sets up a viewer for visualizing the simulation, allowing keyboard interactions.
        """
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'setup_viewer'})
            message = client.recv_pyobj()

    def render(self, sync_frame_time=True):
        """
        Renders the simulation frame-by-frame, syncing frame time if required.

        Args:
            sync_frame_time (bool): Whether to synchronize the frame time.
        """
        for client in self.simulator_clients.values():
            client.send_pyobj({'attr': 'render', 'args': [sync_frame_time]})
            message = client.recv_pyobj()

    
    
    def get_env_counts(self, num_envs):
        ratios = self.config.simulator_ratios
        assert len(ratios) == len(self.config.simulator_names), "Length of simulator_ratios must match number of simulator_names."
        assert abs(sum(ratios) - 1.0) < 1e-5, "simulator_ratios must sum to 1.0"

        env_counts = {name: int(num_envs * ratio) for name, ratio in zip(self.config.simulator_names, ratios)}
        return env_counts

    def separate_env_ids(self, env_ids):
        separated_ids = {}
        for simulator_name in self.config.simulator_names:
            separated_ids[simulator_name] = []

        for env_id in env_ids:
            cumulative_count = 0
            for simulator_name, count in self.env_counts.items():
                if cumulative_count <= env_id < cumulative_count + count:
                    separated_ids[simulator_name].append(env_id - cumulative_count)
                    break
                cumulative_count += count

        for simulator_name in separated_ids:
            separated_ids[simulator_name] = torch.tensor(separated_ids[simulator_name], dtype=torch.long, device=self.sim_device)

        return separated_ids
    

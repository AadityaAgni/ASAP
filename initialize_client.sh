# CUDA_VISIBLE_DEVICES=4 python humanoidverse/train_agent.py \
# +simulator=genesis \
# +exp=locomotion \
# +domain_rand=NO_domain_rand \
# +rewards=loco/reward_h1_12dof_locomotion \
# +robot=g1/g1_12dof \
# +terrain=terrain_locomotion_plane \
# +obs=loco/leggedloco_obs_singlestep_withlinvel \
# num_envs=1 \
# project_name=TESTInstallation \
# experiment_name=G112dof_loco_IsaacGym \
# headless=True


python3 humanoidverse/initialize_client.py \
+exp=locomotion_genesis \
+simulator=genesis \
+domain_rand=domain_rand_base \
+rewards=loco/reward_g1_locomotion \
+robot=g1/g1_29dof_anneal_23dof.yaml \
+terrain=terrain_locomotion_plane \
+obs=loco/leggedloco_obs_singlestep_withlinvel \
num_envs=4096 \
project_name=HumanoidLocomotion \
experiment_name=H110dof_loco_Genesis \
headless=True \
rewards.reward_penalty_curriculum=True \
rewards.reward_initial_penalty_scale=0.1
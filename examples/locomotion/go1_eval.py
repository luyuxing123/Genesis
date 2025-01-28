import argparse
import os
import pickle
import glob
import torch
import threading
import keyboard
from go1_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
import genesis as gs

# Global variables to store speed and angular velocity
lin_x = 0
ang_z = 0

# Function to read keyboard input in a separate thread
def read_input():
    global lin_x, ang_z
    while True:
        user_input = input("Enter command (w/s/a/d): ")
        if user_input == "w":  # Increase speed
            lin_x += 1
        elif user_input == "s":  # Decrease speed
            lin_x -= 1
        elif user_input == "a":  # Rotate right
            ang_z += 1
        elif user_input == "d":  # Rotate left
            ang_z -= 1
        else:
            print("Invalid command. Use 'w', 's', 'a', or 'd'.")

        print(lin_x,ang_z)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go1-walking")
    parser.add_argument("--ckpt", type=int, default=10000)
    args = parser.parse_args()

    dirs = glob.glob(f"logs/go1-walking/*")
    logdir = sorted(dirs)[-1]

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))

    command_cfg = {
        "num_commands": 2,
        "lin_vel_x_range": [lin_x, lin_x],
        "ang_vel_range": [ang_z, ang_z],
    }

    env_cfg["termination_if_roll_greater_than"] = 1000

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(logdir)
    policy = runner.get_inference_policy(device="cuda:0")

    # Start the keyboard input thread
    input_thread = threading.Thread(target=read_input, daemon=True)
    input_thread.start()

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            # Update the observation based on the latest lin_x and ang_z values
            obs[:,6] = lin_x*2
            obs[:,7] = ang_z*0.25
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

if __name__ == "__main__":
    main()

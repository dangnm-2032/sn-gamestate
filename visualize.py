import shutil
import os
import subprocess
import yaml
import argparse

CONFIG_PATH = 'sn_gamestate/configs/soccernet.yaml'
EXTRACTED_FRAME_PATH = 'data/Analyze/valid'

def run(video_path: str, job_id: str, state_path: str | None = None):
    # Check for ffmpeg
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found")
        exit(1)

    # Extract video to images with ffmpeg
    # ffmpeg -i test.mp4 -qscale:v 1 data/Analyze/valid/e2127b0e-0107-481c-a453-6f3a291c6d81/img1/%06d.jpg
    output_dir = os.path.join(EXTRACTED_FRAME_PATH, job_id, 'img1')

    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i", video_path,
        "-qscale:v", "1",
        f"{output_dir}/%06d.jpg"
    ]

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

    # Configure visualization
    # Edit configs/soccernet.yaml to set the video path
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['job_jd'] = job_id
    config['dataset']['vids_dict']['valid'] = [job_id]
    config['hydra']['run']['dir'] = "outputs/${job_jd}"
    config['dataset']['dataset_path'] = r"${data_dir}/Analyze"
    config['eval_tracking'] = False
    config['pipeline'] = []
    config['state']['save_file'] = None
    config['state']['load_file'] = state_path
    config['visualization']['cfg']['save_videos'] = True
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)

    # Run visualization
    # python visualize.py
    subprocess.run(["uv", "run", "tracklab", "-cn", "soccernet"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--job-id", type=str, required=True)
    parser.add_argument("--state", type=str, default=True)
    args = parser.parse_args()
    cwd = os.getcwd()
    run(os.path.join(cwd, args.video), args.job_id, os.path.join(cwd, args.state))

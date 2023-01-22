import os
import sys
import random
import re
import json

import numpy as np

# https://docs.paramiko.org/en/stable/api/client.html
from paramiko import SSHClient, AutoAddPolicy, ssh_exception

def open_connection(hostname, username, ssh_key_path, port):
    """
    If successful, this will connect to the remote server and
    the value of self.ssh_client will be usable.
    Otherwise, this will set self.ssh_client=None or it will quit().
    """

    ssh_client = SSHClient()
    ssh_client.set_missing_host_key_policy(AutoAddPolicy())
    ssh_client.load_system_host_keys()
    assert os.path.exists(
        ssh_key_path
    ), f"Error. The absolute path given for ssh_key_path does not exist: {ssh_key_path} ."

    # The call to .connect was seen to raise an exception now and then.
    #     raise AuthenticationException("Authentication timeout.")
    #     paramiko.ssh_exception.AuthenticationException: Authentication timeout.
    # When it happens, we should simply give up on the attempt
    # and log the error to stdout.
    try:
        # For some reason, we really need to specify which key_filename to use.
        ssh_client.connect(
            hostname, username=username, port=port, key_filename=ssh_key_path
        )
        print(f"Successful SSH connection to {username}@{hostname} port {port}.")
    except ssh_exception.AuthenticationException as inst:
        print(f"Error in SSH connection to {username}@{hostname} port {port}.")
        print(type(inst))
        print(inst)
        # set the ssh_client to None as a way to communicate
        # to the parent that we got into trouble
        ssh_client = None
    except Exception as inst:
        print(f"Error in SSH connection to {username}@{hostname} port {port}.")
        print(type(inst))
        print(inst)
        ssh_client = None

    return ssh_client


def sample_one_experiment_config():
    """
    Selects one particular job configuration at random.
    
    Returns:
        Job configuration in JSON format.
    """
    config = {}
    config['n_tasks'] = random.choice([ 1, 2, 4, 8 ])
    config['time'] = random.choice([ '01:00:00', '02:00:00', '04:00:00', '08:00:00' ])
    config['mem_per_cpu'] = random.choice([ '8Gb', '16Gb' ])
    config['gpus'] = random.choice([ 1, 2 ])
    config['cpus_per_task'] = random.choice([ 1, 2 ])
    
    return config


def run(hostname, username, ssh_key_path, port):
    """Connects to remote server through SSH and launches fake job."""
    
    ssh_client = open_connection(
        hostname, username, ssh_key_path, port)
    assert ssh_client

    unique_id = np.random.randint(low=0, high=1e8)
    output_dir = "sbatch_estimations"  # inside scratch directory
    if hostname != "login.server.mila.quebec":
        account_str = "--account=rrg-bengioy-ad"
    else:
        account_str = ""

    config = sample_one_experiment_config()
    print(config)
    
    redirect_to_file = f"> {output_dir}/{unique_id}_fake_job_sbatch.txt 2>&1"

    remote_cmd = f"""
    mkdir -p scratch
    cd scratch
    mkdir -p {output_dir}
    echo 'Job info: {json.dumps(config)}' {redirect_to_file}
    sbatch --test-only {account_str} -J {unique_id}_fake_job_sbatch -n {config['n_tasks']} -t {config['time']} --mem-per-cpu={config['mem_per_cpu']} -G {config['gpus']} -c {config['cpus_per_task']} --wrap="date" >{redirect_to_file}
    sbatch {account_str} -J {unique_id}_fake_job_sbatch -n {config['n_tasks']} -t {config['time']} --mem-per-cpu={config['mem_per_cpu']} -G {config['gpus']} -c {config['cpus_per_task']} --wrap="(echo 'Actual start time:' && date) >{redirect_to_file}" >{redirect_to_file}
    """

    print('Job unique id:', unique_id)
    # print(remote_cmd)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh_client.exec_command(remote_cmd)

    for line in ssh_stdout.readlines():
        print(line)

    for line in ssh_stderr.readlines():
        print(line)


if __name__ == "__main__":
    assert len(sys.argv) == 3
    assert "@" in sys.argv[1]
    if sys.argv[1].__contains__(":"):
        username, hostname, port = re.split("@|:", sys.argv[1])
    else:
        username, hostname = re.split("@", sys.argv[1])
        port = 22
        
    ssh_key_path = sys.argv[2]
    run(hostname, username, ssh_key_path, port)

"""
For CC cluster:
python3 mila_cc/slurm_start_time_estimation/submit_fake_and_true_job_to_remote_cluster.py alaingui@narval.computecanada.ca ~/.ssh/id_rsa.pub

For mila cluster:
python3 mila_cc/slurm_start_time_estimation/submit_fake_and_true_job_to_remote_cluster.py bianca.popa@login.server.mila.quebec:2222 ~/.ssh/id_rsa.pub
"""
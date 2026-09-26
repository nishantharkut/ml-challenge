"""Helper to set up SSH key auth and run remote commands."""
import subprocess
import sys
import os
import time

REMOTE = "covid@192.168.83.125"
PASSWORD = "covid"

def setup_ssh_key():
    """Generate SSH key if needed and copy to remote."""
    key_path = os.path.expanduser("~/.ssh/id_rsa")
    if not os.path.exists(key_path):
        print("Generating SSH key...")
        subprocess.run(
            ["ssh-keygen", "-t", "rsa", "-N", "", "-f", key_path],
            check=True
        )
    
    pub_key_path = key_path + ".pub"
    with open(pub_key_path, 'r') as f:
        pub_key = f.read().strip()
    
    print(f"Public key: {pub_key[:60]}...")
    print(f"\nPlease run this on the remote machine manually:")
    print(f'  mkdir -p ~/.ssh && echo "{pub_key}" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys')
    return pub_key

def run_remote(cmd, timeout=300):
    """Run command on remote via SSH."""
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
         REMOTE, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode

if __name__ == "__main__":
    setup_ssh_key()

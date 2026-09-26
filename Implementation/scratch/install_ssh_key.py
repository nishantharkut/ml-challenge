"""Install SSH key on remote machine using paramiko."""
import paramiko
import os

pub_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
with open(pub_key_path) as f:
    pub_key = f.read().strip()

print(f"Key length: {len(pub_key)} chars")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.83.125", username="covid", password="covid", timeout=10)
print("Connected!")

# Clear broken keys and install correct one as single line
cmd = f'rm -f ~/.ssh/authorized_keys && mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo "{pub_key}" > ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && wc -l ~/.ssh/authorized_keys && echo KEY_INSTALLED_OK'
stdin, stdout, stderr = client.exec_command(cmd)
out = stdout.read().decode()
err = stderr.read().decode()
print(f"OUT: {out}")
if err:
    print(f"ERR: {err}")
client.close()
print("Done!")

# AWS EC2 High-Speed Inference Guide (Amazon ML Challenge 2026)

This guide walks you through using your $100 AWS credits to run test inference on a 64-core or 32-core EC2 instance in **under 2 hours**.

---

## 1. Instance Specification & Cost

| Instance Type | vCPUs | RAM | Cost / Hour | Estimated Time for US | Total Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`c6i.16xlarge` (Recommended)** | **64** | **128 GB** | **~$2.72 / hr** | **~1.5 hours** | **~$4.00** |
| **`c6i.8xlarge`** | **32** | **64 GB** | **~$1.36 / hr** | **~2.5 hours** | **~$3.50** |

*Both fit easily into your $100 free tier / promotional credits.*

---

## 2. Launch the Instance via AWS Console (5 Minutes)

1. Open the [AWS EC2 Console](https://console.aws.amazon.com/ec2/home).
2. Ensure your region is set to **`us-east-1` (N. Virginia)** or **`ap-south-1` (Mumbai)** in the top right corner.
3. Click **"Launch Instance"**:
   - **Name:** `amazon-ml-runner`
   - **Application and OS Images:** **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type** (64-bit x86).
   - **Instance type:** Select **`c6i.16xlarge`** (or `c6i.8xlarge`).
   - **Key pair (login):**
     - Select an existing key pair or click **"Create new key pair"** (Name: `amazon-ml-key`, format: `.pem` for OpenSSH / Linux / Mac / Windows PowerShell).
     - Save `amazon-ml-key.pem` to your computer (e.g. `C:\Users\nhnis\.ssh\amazon-ml-key.pem`).
   - **Network settings:**
     - Check **"Allow SSH traffic from anywhere (0.0.0.0/0)"** (or My IP).
   - **Configure storage:**
     - Change the root volume size from 8 GiB to **`100 GiB`** (General Purpose SSD `gp3`).
4. Click **"Launch Instance"**.
5. Note the **Public IPv4 address** of the running instance (e.g., `3.85.120.45`).

---

## 3. Connect via SSH

From PowerShell on your local machine:
```powershell
# Set permissions on key (Windows)
icacls "$env:USERPROFILE\.ssh\amazon-ml-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"

# Connect to EC2
ssh -i "$env:USERPROFILE\.ssh\amazon-ml-key.pem" ubuntu@<PUBLIC_IP>
```

---

## 4. Run the 1-Step Setup on the EC2 Instance

Once connected to the EC2 instance, copy and paste this entire block into the terminal:

```bash
# 1. Update and install dependencies (including unzip)
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git htop unzip

# 2. Create virtual environment
python3 -m venv ~/ml-venv
source ~/ml-venv/bin/activate
pip install --upgrade pip
pip install rapidfuzz sparse_dot_topn lightgbm polars anyascii psutil scipy scikit-learn joblib

# 3. Create project structure
mkdir -p ~/amazon_ml/dataset/test
mkdir -p ~/amazon_ml/Implementation/runs/default/checkpoints
mkdir -p ~/amazon_ml/Implementation/runs/default/output

echo "Environment ready!"
```

---

## 5. Transfer Code and Data to the Instance

Run these commands from your **Local Machine (PowerShell)**:

```powershell
$KEY = "$env:USERPROFILE\.ssh\amazon-ml-key.pem"
$IP = "<PUBLIC_IP>"

# 1. Sync code, models, and decoder (1.28 MB bundle)
scp -i $KEY "C:\N Drive\Amazon ML Challenge\ec2_bundle.zip" ubuntu@${IP}:~/amazon_ml/
ssh -i $KEY ubuntu@${IP} "unzip -o ~/amazon_ml/ec2_bundle.zip -d ~/amazon_ml/ && rm ~/amazon_ml/ec2_bundle.zip"

# 2. Sync compressed test dataset (488 MB test_data.zip)
scp -i $KEY "C:\N Drive\Amazon ML Challenge\test_data.zip" ubuntu@${IP}:~/amazon_ml/
ssh -i $KEY ubuntu@${IP} "unzip -o ~/amazon_ml/test_data.zip -d ~/amazon_ml/ && rm ~/amazon_ml/test_data.zip"
```

---

## 6. Execute US Test Inference on EC2 (64 Cores)

On the EC2 instance terminal:

```bash
cd ~/amazon_ml
source ~/ml-venv/bin/activate

export AMAZON_ML_DATASET_DIR="$HOME/amazon_ml/dataset"
export AMAZON_ML_RUNS_DIR="$HOME/amazon_ml/Implementation/runs"
export AMAZON_ML_RUN_ID="default"

# Launch in background with nohup (safe from SSH disconnects!)
nohup python -u Implementation/51_stream_country.py --country US --shard-size 25000 > us_run.log 2>&1 &

# View real-time progress (press Ctrl+C to exit log view anytime; the process will keep running!)
tail -f us_run.log
```

*Estimated completion: **~1.5 hours** on `c6i.16xlarge` (64 cores, 128 GB RAM).*


---

## 7. Retrieve Outputs & Terminate Instance

From your local machine (PowerShell):

```powershell
$KEY = "$env:USERPROFILE\.ssh\amazon-ml-key.pem"
$IP = "<PUBLIC_IP>"

# Download partition results
scp -i $KEY ubuntu@${IP}:~/amazon_ml/Implementation/runs/default/output/part_cands_US.tsv "C:\N Drive\Amazon ML Challenge\Implementation\runs\default\output\"
scp -i $KEY ubuntu@${IP}:~/amazon_ml/Implementation/runs/default/output/part_match_US.tsv "C:\N Drive\Amazon ML Challenge\Implementation\runs\default\output\"

Write-Host "US Partition outputs downloaded successfully!"
```

### IMPORTANT: Terminate Instance
Once downloaded, return to the **AWS EC2 Console**, select the instance, and click:
**Instance state -> Terminate instance**.
*(This stops all billing and preserves your remaining credits!)*

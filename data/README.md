python3 -m venv ~/venvs/broker_benchmark
source ~/venvs/broker_benchmark/bin/activate
cd "/workspace/Sinica/Agentic Memory/broker_benchmark_linux"
pip install -r requirements.txt   # only if the folder has one
python download_broker_evidence.py

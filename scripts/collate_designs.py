import json
import os
import sys


LOG_FILE = "PP_LOG.json"


def check_simulator(dir):
	files = []
	for f in os.listdir(dir):
		
		subdir = os.path.join(dir, f)
		if not os.path.isdir(subdir):
			continue
		
		if os.path.exists(os.path.join(subdir, 'dqn_hparams_hyform.json')):
			files.append(subdir)
		else:
			with open(LOG_FILE, "a") as f:
				f.writelines([json.dumps({"file": subdir, "status": "skipped", "info": "no hyform params"})])
			
	return files
		

def collect_designs():
	
	all_data = []
	for f in check_simulator("./experiments/DQN"):
		keyerrors = 0
		print(f"Processing {f}")
		datafile = os.path.join(f, "datafile_log.json")
		if not os.path.exists(datafile):
			print(f"No datafile for {f}")
			with open(LOG_FILE, "a") as f:
				f.writelines([json.dumps({"file": f, "status": "skipped", "info": "no datafile"})])
			continue
		
		try:
			data = json.load(open(datafile))
		except Exception:
			try:
				with open(datafile, "r") as f:
					data = f.read()
					data = json.loads(f"[{data}]")
			except Exception:
				print(f"Could not read {datafile}")
				continue
				
		for d in data:
			uav_data = dict()
			
			try:
				uav_data["config"] = d["config"]
				uav_data["range"] = d["range"]
				uav_data["cost"] = d["cost"]
				uav_data["velocity"] = d["velocity"]
				uav_data["result"] = d["result"]
			except KeyError as e:
				keyerrors += 1
				continue
			
			all_data.append(uav_data)
			
		with open(LOG_FILE, "a") as f:
			f.writelines([json.dumps({"file": str(f), "status": "processed", "info": ""})])
		print(f"Keyerrors: {keyerrors}/{len(data)}")
	
	print(f"Loaded {len(all_data)} designs")
	with open("raw_designs.json", "w") as f:
		f.write(json.dumps(all_data))
		

if __name__ == "__main__":
	import pandas as pd
	import numpy as np
	# collect_designs()
	# df = pd.read_json("raw_designs.json")
	# df.to_csv("raw_designs.csv", index=False)
	
	df = pd.read_csv("raw_designs.csv")
	
	print(f'{np.sum((df["result"] == "Success").values)}/{len(df)} successful designs')

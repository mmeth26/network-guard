# Network Guard
Network is a local Python and Streamlit app for exploring network-flow data. I built it to learn how machine learning models are used and evaluated in cybersecurity.

## What I built 
- Two modes: open a scored CSV or analyze an unscored CSV with saved models.
- A dashboard that compares Random Forest, Isolation Forest, and a simple rule. 
- A review table that sorts flagged flows by score, filters by traffic label when available, and exports the filtered results. 
- Input checks and score caching so changing a filter doesn't rerun the models. 

## How to run it
From the project folder, activiate the environment and start the app:

```bash
source .venv/bin/activate
python3 -m streamlit run app.py --server.maxUploadSize 400
```

Open `http://localhost:8501`. For a quick demo, choose **View scored results**
and upload `v3_wednesday_frozen_run1/scored_flows.csv`. To analyze an
unscored CSV, the app needs `v3_grouped_run1/models.joblib`.

## What I learned 
I evaluated the detectors using CICID2017 network-flow data. Random Forest flagged 231 of 283 attack flows in the held-out Tuesday development splits. With the model and threshold unchanged, it flagged 0 of 18,248 attack flows in the Wednesday sample, which contained different attack types. 

A strong result on familiar attacks didn't carry over to different ones. The app flags flow for review. It does not confirm attacks or monitor live traffic. The app reads up to the first 50,000 rows of an upload; the Wednesday benchmark used a random sample.

## Planned Improvement
- Train on a wider range of attack types and test on a seperate capture to see whether detection improves on unfamiliar traffic.
- Add a small synthetic CSV so anyone can try the dashboard without downloading the full dataset or training a model.

## Credit
I started from [Melisa Sever's threat detection demo](https://github.com/melisasvr/AI-Powered-Cybersecurity-Threat-Detection-System).
Network Guard adds a separate network-flow evaluation pipeline and Streamlit
app alongside the original demo.
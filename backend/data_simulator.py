import argparse, random, csv, time, os
from datetime import datetime, timedelta

def simulate(rows: int, out: str, sleep: float = 0.0):
    # Create folder if it doesn't exist
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fieldnames = ["timestamp","device_id","temp_c","vibration","power_kw","occupancy"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    start = datetime.utcnow()
    for i in range(rows):
        ts = start + timedelta(seconds=i)
        record = {
            "timestamp": ts.isoformat(),
            "device_id": f"AHU-{1 + (i % 3)}",
            "temp_c": round(22 + random.gauss(0, 0.8) + 0.01*i/rows*5, 2),
            "vibration": round(0.2 + 0.005*(i/rows) + abs(random.gauss(0, 0.02)), 3),
            "power_kw": round(15 + random.gauss(0, 1.5) + 0.005*(i/rows)*10, 2),
            "occupancy": int(max(0, 50 + 30*random.random())),
        }
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(record)
        if sleep:
            time.sleep(sleep)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--out", type=str, default="data/sensor_stream.csv")
    p.add_argument("--sleep", type=float, default=0.0)
    a = p.parse_args()
    simulate(a.rows, a.out, a.sleep)
    print(f"Wrote {a.rows} rows to {a.out}")
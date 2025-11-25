import glob, json, datetime
import matplotlib.pyplot as plt

# Récupérer tous les résumés FBA
summaries = []
for path in glob.glob("data/*_FBA_*_summary.json"):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    dt = datetime.datetime.strptime(s["timestamp"], "%Y%m%d_%H%M%S")
    summaries.append((dt, s["final_threshold_deg"], s["accuracy_percent"]))

# Trier par date
summaries.sort(key=lambda x: x[0])

dates = [s[0] for s in summaries]
thresholds = [s[1] for s in summaries]
acc = [s[2] for s in summaries]

# Courbe du seuil (c'est le plus important)
plt.figure()
plt.plot(dates, thresholds, marker="o")
plt.gca().invert_yaxis()  # plus bas = mieux
plt.xlabel("Session")
plt.ylabel("Direction threshold (deg)")
plt.title("FBA training – direction threshold over sessions")

# Optionnel : courbe d'accuracy
plt.figure()
plt.plot(dates, acc, marker="o")
plt.xlabel("Session")
plt.ylabel("Accuracy (%)")
plt.title("FBA training – accuracy over sessions")

plt.show()

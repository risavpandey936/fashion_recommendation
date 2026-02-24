import pickle
import os

with open("filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

fixed = [
    os.path.join("images", os.path.basename(p))
    for p in filenames
]

with open("filenames.pkl", "wb") as f:
    pickle.dump(fixed, f)

print("✅ filenames.pkl fixed for local usage")
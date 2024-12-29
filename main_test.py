import requests
import os
path=os.path.dirname(__file__)
url = "https://huggingface.co/datasets/datnguyentien204/ViLC-EVJVQA-Datasets/resolve/main/uit-vilc-efficientnet-b0.pt"
r = requests.get(url, allow_redirects=True)
os.makedirs(os.path.join(path, 'static/weights'), exist_ok=True)
open(os.path.join(path, 'static/weights/uit-vilc-efficientnet-b0.pt'), 'wb').write(r.content)
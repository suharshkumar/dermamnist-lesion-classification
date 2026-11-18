# Figure for the paper: one representative test image per class at 28, 64, 128 and 224 px (same image index per row),
# to make the resolution argument visually. Writes /content/DDP_Phase3/ensemble/class_examples.png (and a 2-column PDF).
import os, urllib.request, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
CLASS_NAMES = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
files = {28: '/content/dermamnist.npz', 64: '/content/dermamnist_64.npz', 128: '/content/dermamnist_128.npz', 224: '/content/dermamnist_224.npz'}
for res, f in files.items():
    if not os.path.exists(f):
        name = 'dermamnist.npz' if res == 28 else f'dermamnist_{res}.npz'
        urllib.request.urlretrieve(f'https://zenodo.org/records/10519652/files/{name}?download=1', f)
arrs = {res: np.load(f) for res, f in files.items()}
y = arrs[224]['test_labels'].flatten()
rng = np.random.default_rng(3)
idx = [int(rng.choice(np.where(y == c)[0])) for c in range(7)]
fig, axes = plt.subplots(7, 4, figsize=(6.2, 10.6))
for r, c in enumerate(range(7)):
    for col, res in enumerate([28, 64, 128, 224]):
        ax = axes[r, col]; ax.imshow(arrs[res]['test_images'][idx[c]], interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(f'{res}×{res} px', fontsize=9)
        if col == 0: ax.set_ylabel(CLASS_NAMES[c], fontsize=8, rotation=0, ha='right', va='center', labelpad=4)
plt.tight_layout(w_pad=0.3, h_pad=0.3)
out = '/content/DDP_Phase3/ensemble'; os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, 'class_examples.png'), dpi=200, bbox_inches='tight'); plt.savefig(os.path.join(out, 'class_examples.pdf'), bbox_inches='tight')
print('saved class_examples; indices', idx)

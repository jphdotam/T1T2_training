import os
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter
from lib.plots import mean_diff_plot
from lib.vis import iou, dice
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import r2_score, accuracy_score, cohen_kappa_score, recall_score, precision_score

from lib.windows import T1_CUTOFF


PATH_AI = "./data/predictions"
PATH_JH = "./data/predictions_james"
PATH_HX = "./data/predictions_hui"

MEAN_OR_MEDIAN = 'median'

with open(os.path.join(PATH_AI, 'segments.json')) as f:
    preds_ai = json.load(f)
with open(os.path.join(PATH_JH, 'segments.json')) as f:
    preds_jh = json.load(f)
with open(os.path.join(PATH_HX, 'segments.json')) as f:
    preds_hx = json.load(f)

for study_id, study_dict in preds_ai.copy().items():
    for segment_id, segment_dict in study_dict.items():
        if np.isnan(segment_dict['mean']):
            print(f"Segmentation fail for {study_id}")
            del preds_ai[study_id]
            break

valid_sequences = list(set(preds_ai.keys()).intersection(preds_jh.keys(), preds_hx.keys()))
print(f"Found {len(valid_sequences)} across the 3")

print(f"{len([k for k in preds_ai if k not in preds_jh and k not in preds_hx])} in AI but missing from both human")
print(f"{len([k for k in preds_ai if k in preds_jh and k not in preds_hx])} in AI but missing from JH only")
print(f"{len([k for k in preds_ai if k not in preds_jh and k  in preds_hx])} in AI but missing from HX only")
print(f"{len([k for k in preds_jh if k not in preds_ai and k in preds_hx])} in AI and HX; not in AI only")
print(f"{len([k for k in preds_jh if k not in preds_ai and k not in preds_hx])} in JH only; not in AI or HX")
print(f"{len([k for k in preds_hx if k not in preds_ai and k not in preds_jh])} in HX only; not in AI or JH")

stacks = list(set([s.split('__')[0] for s in valid_sequences]))
print(f"Found {len(stacks)} stacks")

patients = list(set([s.split('_')[2] for s in stacks]))
print(f"Found {len(patients)} patients")

###################### mean

blood_t1s_segwise = []
blood_t1_by_seq = {}

######
# t1 #
######

ai_t1, jh_t1, hx_t1 = [], [], []
for study in valid_sequences:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t1, jh_t1, hx_t1)):
        for segment_id, segment_dict in preds[study].items():
            if 't1' in segment_id:
                list_to.append(segment_dict[MEAN_OR_MEDIAN])
                if 'lvt1' in segment_dict:
                    blood_t1s_segwise.append(segment_dict['lvt1'])
                    blood_t1_by_seq[study] = segment_dict['lvt1']


ai_t1 = np.array(ai_t1)
jh_t1 = np.array(jh_t1)
hx_t1 = np.array(hx_t1)

ai_t1_native = np.array([seg_t1 for seg_t1, blood_t1 in zip(ai_t1, blood_t1s_segwise) if blood_t1 >= T1_CUTOFF])
jh_t1_native = np.array([seg_t1 for seg_t1, blood_t1 in zip(jh_t1, blood_t1s_segwise) if blood_t1 >= T1_CUTOFF])
hx_t1_native = np.array([seg_t1 for seg_t1, blood_t1 in zip(hx_t1, blood_t1s_segwise) if blood_t1 >= T1_CUTOFF])

ai_t1_gad = np.array([seg_t1 for seg_t1, blood_t1 in zip(ai_t1, blood_t1s_segwise) if blood_t1 < T1_CUTOFF])
jh_t1_gad = np.array([seg_t1 for seg_t1, blood_t1 in zip(jh_t1, blood_t1s_segwise) if blood_t1 < T1_CUTOFF])
hx_t1_gad = np.array([seg_t1 for seg_t1, blood_t1 in zip(hx_t1, blood_t1s_segwise) if blood_t1 < T1_CUTOFF])

ai_t2, jh_t2, hx_t2 = [], [], []
for study in valid_sequences:
    for preds, list_to in zip((preds_ai, preds_jh, preds_hx), (ai_t2, jh_t2, hx_t2)):
        # for preds, list_to in zip((preds_ai, preds_hx), (ai_t2, hx_t2)):
        for segment_id, segment_dict in preds[study].items():
            if 't2' in segment_id:
                list_to.append(segment_dict[MEAN_OR_MEDIAN])

ai_t2 = np.array(ai_t2)
jh_t2 = np.array(jh_t2)
hx_t2 = np.array(hx_t2)

mae_jh_ai_t1, mae_hx_ai_t1, mae_jh_hx_t1, mae_jh_ai_t2, mae_hx_ai_t2, mae_jh_hx_t2 = [], [], [], [], [], [],
for jt1, jt2, ht1, ht2, at1, at2 in zip(jh_t1, jh_t2, hx_t1, hx_t2, ai_t1, ai_t2):
    mae_jh_ai_t1.append(abs(jt1 - at1))
    mae_hx_ai_t1.append(abs(ht1 - at1))
    mae_jh_hx_t1.append(abs(jt1 - ht1))
    mae_jh_ai_t2.append(abs(jt2 - at2))
    mae_hx_ai_t2.append(abs(ht2 - at2))
    mae_jh_hx_t2.append(abs(jt2 - ht2))
mae_jh_ai_t1_native, mae_hx_ai_t1_native, mae_jh_hx_t1_native = [], [], []
for jt1n, ht1n, at1n in zip(jh_t1_native, hx_t1_native, ai_t1_native):
    mae_jh_ai_t1_native.append(abs(jt1n - at1n))
    mae_hx_ai_t1_native.append(abs(ht1n - at1n))
    mae_jh_hx_t1_native.append(abs(jt1n - ht1n))
mae_jh_ai_t1_gad, mae_hx_ai_t1_gad, mae_jh_hx_t1_gad = [], [], []
for jt1n, ht1n, at1n in zip(jh_t1_gad, hx_t1_gad, ai_t1_gad):
    mae_jh_ai_t1_gad.append(abs(jt1n - at1n))
    mae_hx_ai_t1_gad.append(abs(ht1n - at1n))
    mae_jh_hx_t1_gad.append(abs(jt1n - ht1n))

spearman_jh_ai_t1_native = spearmanr(jh_t1_native, ai_t1_native).correlation
spearman_hx_ai_t1_native = spearmanr(hx_t1_native, ai_t1_native).correlation
spearman_jh_hx_t1_native = spearmanr(jh_t1_native, hx_t1_native).correlation
spearman_jh_ai_t1_gad = spearmanr(jh_t1_gad, ai_t1_gad).correlation
spearman_hx_ai_t1_gad = spearmanr(hx_t1_gad, ai_t1_gad).correlation
spearman_jh_hx_t1_gad = spearmanr(jh_t1_gad, hx_t1_gad).correlation
spearman_jh_ai_t2 = spearmanr(jh_t2, ai_t2).correlation
spearman_hx_ai_t2 = spearmanr(hx_t2, ai_t2).correlation
spearman_jh_hx_t2 = spearmanr(jh_t2, hx_t2).correlation

try:
    r2_jh_ai_t1_native = r2_score(jh_t1_native, ai_t1_native)
    r2_hx_ai_t1_native = r2_score(hx_t1_native, ai_t1_native)
    r2_jh_hx_t1_native = r2_score(jh_t1_native, hx_t1_native)
    r2_jh_ai_t1_gad = r2_score(jh_t1_gad, ai_t1_gad)
    r2_hx_ai_t1_gad = r2_score(hx_t1_gad, ai_t1_gad)
    r2_jh_hx_t1_gad = r2_score(jh_t1_gad, hx_t1_gad)
except ValueError:
    print(f"LV T1 not known")
r2_jh_ai_t2 = r2_score(jh_t2, ai_t2)
r2_hx_ai_t2 = r2_score(hx_t2, ai_t2)
r2_jh_hx_t2 = r2_score(jh_t2, hx_t2)

mae_jh_ai_t1 = np.mean(mae_jh_ai_t1)
mae_hx_ai_t1 = np.mean(mae_hx_ai_t1)
mae_jh_hx_t1 = np.mean(mae_jh_hx_t1)
mae_jh_ai_t2 = np.mean(mae_jh_ai_t2)
mae_hx_ai_t2 = np.mean(mae_hx_ai_t2)
mae_jh_hx_t2 = np.mean(mae_jh_hx_t2)
mae_jh_ai_t1_native = np.mean(mae_jh_ai_t1_native)
mae_hx_ai_t1_native = np.mean(mae_hx_ai_t1_native)
mae_jh_hx_t1_native = np.mean(mae_jh_hx_t1_native)
mae_jh_ai_t1_gad = np.mean(mae_jh_ai_t1_gad)
mae_hx_ai_t1_gad = np.mean(mae_hx_ai_t1_gad)
mae_jh_hx_t1_gad = np.mean(mae_jh_hx_t1_gad)

def add_jitter(values, max_jitter):
    return np.array([v + np.random.uniform(-max_jitter, max_jitter, 1)[0] for v in values])


###############
# t1 combined #
###############

fig_t1, axes_t1_comb = plt.subplots(2, 3, sharey='row', figsize=(15, 6))
for i, (ax, title) in enumerate(
        zip(axes_t1_comb[0], ('\nAI versus Expert 1', '\nAI versus Expert 2', '\nExpert 1 versus Expert 2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.yaxis.set_ticks(np.arange(0, 2000, 400))
    ax.xaxis.set_ticks(np.arange(0, 2000, 400))

for i, ax in enumerate(axes_t1_comb[1]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 2000, 400))
    ax.yaxis.set_ticks(np.arange(-800, 1800, 400))

colours = ['C1' if t1 < T1_CUTOFF else 'C0' for t1 in blood_t1s_segwise]

axes_t1_comb[0, 0].scatter(x=jh_t1, y=ai_t1, alpha=0.1, c=colours)
axes_t1_comb[0, 1].scatter(x=hx_t1, y=ai_t1, alpha=0.1, c=colours)
axes_t1_comb[0, 2].scatter(x=jh_t1, y=hx_t1, alpha=0.1, c=colours)
for axis in axes_t1_comb[0]:
    axis.set_xlabel("T1 (ms)", fontsize=15)
    axis.set_ylabel("T1 (ms)", fontsize=15)

axes_t1_comb[0, 0].plot([0, 2000], [0, 2000], 'k--')
axes_t1_comb[0, 1].plot([0, 2000], [0, 2000], 'k--')
axes_t1_comb[0, 2].plot([0, 2000], [0, 2000], 'k--')

mean_diff_plot(ai_t1, jh_t1, ax=axes_t1_comb[1, 0], scatter_kwds={'alpha': 0.1, 'c': colours}, mae=np.round(mae_jh_ai_t1, 1))
mean_diff_plot(ai_t1, hx_t1, ax=axes_t1_comb[1, 1], scatter_kwds={'alpha': 0.1, 'c': colours}, mae=np.round(mae_hx_ai_t1, 1))
mean_diff_plot(jh_t1, hx_t1, ax=axes_t1_comb[1, 2], scatter_kwds={'alpha': 0.1, 'c': colours}, mae=np.round(mae_jh_hx_t1, 1))

plt.setp(axes_t1_comb, xlim=(0, 1800))
plt.setp(axes_t1_comb[0], ylim=(0, 1700))
plt.setp(axes_t1_comb[1], ylim=(-900, 900))
plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.075)
fig_t1.suptitle('T1 (ms) - Scatter and Bland-Altman plots comparing AI & expert measurements\n', fontsize=20)

###############
# t1 separate #
###############

fig_t1, axes_t1_sep = plt.subplots(4, 3, sharey='row', figsize=(15, 6))
for i, (ax, title) in enumerate(
        zip(axes_t1_sep[0], ('\nAI versus Expert 1', '\nAI versus Expert 2', '\nExpert 1 versus Expert 2'))):
    ax.set_title(title, fontsize=12)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.yaxis.set_ticks(np.arange(0, 2000, 400))
    ax.xaxis.set_ticks(np.arange(0, 2000, 400))

for i, (ax, title) in enumerate(
        zip(axes_t1_sep[1], ('\nAI versus Expert 1', '\nAI versus Expert 2', '\nExpert 1 versus Expert 2'))):
    # ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.yaxis.set_ticks(np.arange(0, 2000, 400))
    ax.xaxis.set_ticks(np.arange(0, 2000, 400))

for i, ax in enumerate(axes_t1_sep[2]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 2000, 400))
    ax.yaxis.set_ticks(np.arange(-400, 1200, 400))

for i, ax in enumerate(axes_t1_sep[3]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 2000, 400))
    ax.yaxis.set_ticks(np.arange(-400, 1200, 400))

# pre
axes_t1_sep[0, 0].scatter(x=jh_t1_native, y=ai_t1_native, alpha=0.1)
axes_t1_sep[0, 1].scatter(x=hx_t1_native, y=ai_t1_native, alpha=0.1)
axes_t1_sep[0, 2].scatter(x=jh_t1_native, y=hx_t1_native, alpha=0.1)
for axis in axes_t1_sep[0]:
    axis.set_xlabel("Native T1 (ms)", fontsize=12)
    axis.set_ylabel("Native T1 (ms)", fontsize=12)

axes_t1_sep[0, 0].plot([0, 2000], [0, 2000], 'k--')
axes_t1_sep[0, 1].plot([0, 2000], [0, 2000], 'k--')
axes_t1_sep[0, 2].plot([0, 2000], [0, 2000], 'k--')
axes_t1_sep[2, 0].plot([0, 2000], [0, 2000], 'k--')
axes_t1_sep[2, 1].plot([0, 2000], [0, 2000], 'k--')
axes_t1_sep[2, 2].plot([0, 2000], [0, 2000], 'k--')

mean_diff_plot(ai_t1_native, jh_t1_native, ax=axes_t1_sep[1, 0], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_ai_t1, 1), axistitle_fontsize=12)
mean_diff_plot(ai_t1_native, hx_t1_native, ax=axes_t1_sep[1, 1], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_hx_ai_t1_native, 1), axistitle_fontsize=12)
mean_diff_plot(jh_t1_native, hx_t1_native, ax=axes_t1_sep[1, 2], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_hx_t1_native, 1), axistitle_fontsize=12)

# post
axes_t1_sep[2, 0].scatter(x=jh_t1_gad, y=ai_t1_gad, alpha=0.1)
axes_t1_sep[2, 1].scatter(x=hx_t1_gad, y=ai_t1_gad, alpha=0.1)
axes_t1_sep[2, 2].scatter(x=jh_t1_gad, y=hx_t1_gad, alpha=0.1)
for axis in axes_t1_sep[2]:
    axis.set_xlabel("Post-contrast T1 (ms)", fontsize=12)
    axis.set_ylabel("Post-contrast T1 (ms)", fontsize=12)

axes_t1_sep[2, 0].plot([2, 2000], [2, 2000], 'k--')
axes_t1_sep[2, 1].plot([2, 2000], [2, 2000], 'k--')
axes_t1_sep[2, 2].plot([2, 2000], [2, 2000], 'k--')

mean_diff_plot(ai_t1_gad, jh_t1_gad, ax=axes_t1_sep[3, 0], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_ai_t1, 1), axistitle_fontsize=12)
mean_diff_plot(ai_t1_gad, hx_t1_gad, ax=axes_t1_sep[3, 1], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_hx_ai_t1_gad, 1), axistitle_fontsize=12)
mean_diff_plot(jh_t1_gad, hx_t1_gad, ax=axes_t1_sep[3, 2], scatter_kwds={'alpha': 0.1}, mae=np.round(mae_jh_hx_t1_gad, 1), axistitle_fontsize=12)

# common

plt.setp(axes_t1_sep[0], xlim=(800,1800), ylim=(800, 1800))
plt.setp(axes_t1_sep[1], xlim=(800,1800), ylim=(-500, 500))
plt.setp(axes_t1_sep[2], xlim=(0,1000), ylim=(0, 1000))
plt.setp(axes_t1_sep[3], xlim=(0,1000), ylim=(-500, 500))
plt.subplots_adjust(wspace=0.1, hspace=0.5, left=0.1, top=0.9, bottom=0.05)
fig_t1.suptitle('Native and post-contrast T1 (ms)\nScatter and Bland-Altman plots comparing AI & expert measurements\n', fontsize=16)

######
# T2 #
######


fig_t2, axes_t2 = plt.subplots(2, 3, sharey='row', figsize=(15, 6))
for i, (ax, title) in enumerate(zip(axes_t2[0], ('\nAI versus Expert 1', '\nAI versus Expert 2', '\nExpert 1 versus Expert 2'))):
    ax.set_title(title, fontsize=16)
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticks(np.arange(0, 100, 20))

for i, ax in enumerate(axes_t2[1]):
    if i > 0:
        ax.axes.get_yaxis().set_visible(False)

    ax.xaxis.set_ticks(np.arange(0, 100, 20))
    ax.yaxis.set_ticks(np.arange(-40, 60, 20))

axes_t2[0, 0].scatter(x=jh_t2, y=ai_t2, alpha=0.1)
axes_t2[0, 1].scatter(x=hx_t2, y=ai_t2, alpha=0.1)
axes_t2[0, 2].scatter(x=jh_t2, y=hx_t2, alpha=0.1)
for axis in axes_t2[0]:
    axis.set_xlabel("T2 (ms)", fontsize=15)
    axis.set_ylabel("T2 (ms)", fontsize=15)


axes_t2[0, 0].plot([0, 100], [0, 100], 'k--')
axes_t2[0, 1].plot([0, 100], [0, 100], 'k--')
axes_t2[0, 2].plot([0, 100], [0, 100], 'k--')

mean_diff_plot(ai_t2, jh_t2, ax=axes_t2[1, 0], scatter_kwds={'alpha': 0.05}, mae=np.round(mae_jh_ai_t2, 2))
mean_diff_plot(ai_t2, hx_t2, ax=axes_t2[1, 1], scatter_kwds={'alpha': 0.05}, mae=np.round(mae_hx_ai_t2, 2))
mean_diff_plot(jh_t2, hx_t2, ax=axes_t2[1, 2], scatter_kwds={'alpha': 0.05}, mae=np.round(mae_jh_hx_t2, 2))

plt.setp(axes_t2, xlim=(30, 80))
plt.setp(axes_t2[0], ylim=(30, 80))
plt.setp(axes_t2[1], ylim=(-25, 25))
plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.075)
fig_t2.suptitle('T2 (ms) - Scatter and Bland-Altman plot comparing AI & expert measurements\n', fontsize=20)

iou_ai_jh, iou_ai_hx, iou_jh_hx = [], [], []
dice_ai_jh, dice_ai_hx, dice_jh_hx = [], [], []
for study in tqdm(valid_sequences):
    ai_seg = skimage.io.imread(os.path.join(PATH_AI, study + '.png'))
    jh_seg = skimage.io.imread(os.path.join(PATH_JH, study + '.png'))
    hx_seg = skimage.io.imread(os.path.join(PATH_HX, study + '.png'))

    iou_ai_jh.append(iou(ai_seg, jh_seg))
    iou_ai_hx.append(iou(ai_seg, hx_seg))
    iou_jh_hx.append(iou(jh_seg, hx_seg))
    dice_ai_jh.append(dice(ai_seg, jh_seg))
    dice_ai_hx.append(dice(ai_seg, hx_seg))
    dice_jh_hx.append(dice(jh_seg, hx_seg))

# Annotations

axes_t1_comb[0, 0].annotate(f"Native ρ {spearman_jh_ai_t1_native:.2f}\nNative R² {r2_jh_ai_t1_native:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_comb[0, 1].annotate(f"Native ρ {spearman_hx_ai_t1_native:.2f}\nNative R² {r2_hx_ai_t1_native:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_comb[0, 2].annotate(f"Native ρ {spearman_jh_hx_t1_native:.2f}\nNative R² {r2_jh_hx_t1_native:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

axes_t1_comb[0, 0].annotate(f"Contrast ρ {spearman_jh_ai_t1_gad:.2f}\nContrast R² {r2_jh_ai_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_comb[0, 1].annotate(f"Contrast ρ {spearman_hx_ai_t1_gad:.2f}\nContrast R² {r2_hx_ai_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_comb[0, 2].annotate(f"Contrast ρ {spearman_jh_hx_t1_gad:.2f}\nContrast R² {r2_jh_hx_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

axes_t1_sep[0, 0].annotate(f"Native ρ {spearman_jh_ai_t1_native:.2f}\nNative R² {r2_jh_ai_t1_native:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_sep[0, 1].annotate(f"Native ρ {spearman_hx_ai_t1_native:.2f}\nNative R² {r2_hx_ai_t1_native:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_sep[0, 2].annotate(f"Native ρ {spearman_jh_hx_t1_native:.2f}\nNative R² {r2_jh_hx_t1_native:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

axes_t1_sep[2, 0].annotate(f"Contrast ρ {spearman_jh_ai_t1_gad:.2f}\nContrast R² {r2_jh_ai_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_sep[2, 1].annotate(f"Contrast ρ {spearman_hx_ai_t1_gad:.2f}\nContrast R² {r2_hx_ai_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t1_sep[2, 2].annotate(f"Contrast ρ {spearman_jh_hx_t1_gad:.2f}\nContrast R² {r2_jh_hx_t1_gad:.2f}", xy=(0.99, 0.4), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

axes_t2[0, 0].annotate(f"ρ {spearman_jh_ai_t2:.2f}\nR² {r2_jh_ai_t2:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t2[0, 1].annotate(f"ρ {spearman_hx_ai_t2:.2f}\nR² {r2_hx_ai_t2:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')
axes_t2[0, 2].annotate(f"ρ {spearman_jh_hx_t2:.2f}\nR² {r2_jh_hx_t2:.2f}", xy=(0.99, 0.6), horizontalalignment='right', verticalalignment='center', fontsize=10, xycoords='axes fraction')

fig_t1.show()
fig_t2.show()

# Outliers
deltas_t1, deltas_t2 = defaultdict(list), defaultdict(list)
for study in valid_sequences:
        preds_study = preds_ai[study]
        for segment_id, segment_dict in preds_study.items():
            if 't1' in segment_id:
                delta = preds_study[segment_id][MEAN_OR_MEDIAN] - np.mean([preds_jh[study][segment_id][MEAN_OR_MEDIAN],
                                                                          preds_hx[study][segment_id][MEAN_OR_MEDIAN]])
                deltas_t1[study].append(delta)
            elif 't2' in segment_id:
                delta = preds_study[segment_id][MEAN_OR_MEDIAN] - np.mean(
                    [preds_jh[study][segment_id][MEAN_OR_MEDIAN],
                     preds_hx[study][segment_id][MEAN_OR_MEDIAN]])
                deltas_t2[study].append(delta)

# for study, ai1, jh1, hx1, ai2, jh2, hx2 in zip(studies_valid, ai_t1, jh_t1, hx_t1, ai_t2, jh_t2, hx_t2):
#     deltas_t1[study].append(ai1 - np.mean([jh1, hx1]))
#     deltas_t2[study].append(ai2 - np.mean([jh2, hx2]))

deltas_t1 = dict(sorted(deltas_t1.items(), key=lambda item: np.mean(np.abs(item[1])), reverse=True))
deltas_t2 = dict(sorted(deltas_t2.items(), key=lambda item: np.mean(np.abs(item[1])), reverse=True))

print(list(deltas_t2.keys())[:10])


# Cases with lots of disease
sd_t1s, sd_t2s = {}, {}
for study in valid_sequences:
    sd_t1s[study] = np.std([sdict[MEAN_OR_MEDIAN] for sid, sdict in preds_hx[study].items() if 't1' in sid])
    sd_t2s[study] = np.std([sdict[MEAN_OR_MEDIAN] for sid, sdict in preds_hx[study].items() if 't2' in sid])

sd_t1s = dict(sorted(sd_t1s.items(), key=lambda item: item[1], reverse=True))
sd_t2s = dict(sorted(sd_t2s.items(), key=lambda item: item[1], reverse=True))


# TABLE 1
unique_patients = set([p.split('_')[2] for p in valid_sequences])
invalid_sequences = [p for p in preds_ai.keys() if p not in valid_sequences]
slice_locations = Counter([os.path.splitext(f)[0].rsplit('_',1)[-1] for f in valid_sequences])
seq_is_native = [True if v >= T1_CUTOFF else False for k, v in blood_t1_by_seq.items()]

unique_native_patients = set([s.split('_')[2] for s,n in zip(valid_sequences, seq_is_native) if n is True])
slice_locations_native = Counter([os.path.splitext(f)[0].rsplit('_',1)[-1] for f,n in zip(valid_sequences, seq_is_native) if n is True])

unique_contrast_patients = set([s.split('_')[2] for s,n in zip(valid_sequences, seq_is_native) if n is False])
slice_locations_contrast = Counter([os.path.splitext(f)[0].rsplit('_',1)[-1] for f,n in zip(valid_sequences, seq_is_native) if n is False])

print(f"Sequences: {len(valid_sequences)}")
print(f"Patients: {len(unique_patients)}")
print(f"Invalid seqs: {len(invalid_sequences)}")
print(f"Native maps: {sum(seq_is_native)}")
print(f"Slice locations:\n{chr(10).join(f'{chr(9)}{k}: {v}' for k,v in slice_locations.items())}")

print(f"Unique native patients: {len(unique_native_patients)}")
print(f"Native slice locations:\n{chr(10).join(f'{chr(9)}{k}: {v}' for k,v in slice_locations_native.items())}")

print(f"Unique contrast patients: {len(unique_contrast_patients)}")
print(f"Contrast slice locations:\n{chr(10).join(f'{chr(9)}{k}: {v}' for k,v in slice_locations_contrast.items())}")



# cutoffs
metrics = {'accu': (accuracy_score, False),
           'kappa': (cohen_kappa_score, False),
           'sens': (recall_score, False),
           'spec': (recall_score, True),
           'ppv': (precision_score, False),
           'npv': (precision_score, True)}

# for cutoff, corrects in correct_jh_ai_native.items():
for cutoff in (1200, 1250, 1300):
    print(f"Native T1 @ {cutoff}")
    print(f"% abn by E1: {np.mean(jh_t1_native > cutoff):.3f}\t{np.mean(hx_t1_native > cutoff):.3f}\t{np.mean(ai_t1_native > cutoff):.3f}")
    for met_name, (met_func, invert) in metrics.items():
        res_jh_ai = met_func(jh_t1_native <= cutoff, ai_t1_native <= cutoff) if invert else met_func(jh_t1_native > cutoff, ai_t1_native > cutoff)
        res_hx_ai = met_func(hx_t1_native <= cutoff, ai_t1_native <= cutoff) if invert else met_func(hx_t1_native > cutoff, ai_t1_native > cutoff)
        res_jh_hx = met_func(jh_t1_native <= cutoff, hx_t1_native <= cutoff) if invert else met_func(jh_t1_native > cutoff, hx_t1_native > cutoff)
        print(f"{met_name.upper():<10}{res_jh_ai:.3f}\t{res_hx_ai:.3f}\t{res_jh_hx:.3f}")

for cutoff in (52, 55, 58):
    print(f"T2 @ {cutoff}")
    print(f"% abn: {np.mean(jh_t2 > cutoff):.3f}\t{np.mean(hx_t2 > cutoff):.3f}\t{np.mean(ai_t2 > cutoff):.3f}")
    for met_name, (met_func, invert) in metrics.items():
        res_jh_ai = met_func(jh_t2 <= cutoff, ai_t2 <= cutoff) if invert else met_func(jh_t2 > cutoff, ai_t2 > cutoff)
        res_hx_ai = met_func(hx_t2 <= cutoff, ai_t2 <= cutoff) if invert else met_func(hx_t2 > cutoff, ai_t2 > cutoff)
        res_jh_hx = met_func(jh_t2 <= cutoff, hx_t2 <= cutoff) if invert else met_func(jh_t2 > cutoff, hx_t2 > cutoff)
        print(f"{met_name.upper():<10}{res_jh_ai:.3f}\t{res_hx_ai:.3f}\t{res_jh_hx:.3f}")

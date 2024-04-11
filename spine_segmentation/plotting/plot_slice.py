from collections import Counter
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt

from spine_segmentation.instance_separation.instance_separation import separate_rois_with_labels
from spine_segmentation.spine_types.labels import get_labels_for_n_classes
from spine_segmentation.visualisation.color_palettes import get_alternating_palette, get_wk_bs_palette


def is_completely_correct(prediction, gt):
    correct_no_edges = True
    ids_correct = Counter()
    for curr, count in zip(*np.unique(gt, return_counts=True)):
        if curr == 0:
            continue
        if count < 20:
            continue

        gt_mask = gt == curr
        occurrence = np.bincount(prediction[gt_mask])
        occurrence[0] = 0
        second_most_common = occurrence.argmax()
        if second_most_common != curr:
            correct_no_edges = False
        ids_correct[curr] += second_most_common == curr
    return correct_no_edges, ids_correct


def get_stats(prediction, gt):
    print(np.unique(prediction), np.unique(gt))
    prediction = torch.tensor(prediction.copy())
    gt = torch.tensor(gt.copy())

    remove_px = 15
    zero_layers = sum(torch.amin(prediction == 0, dim=(0, 1)))
    top_remove_px = remove_px + zero_layers
    print(f"{remove_px=}, {top_remove_px=}")

    appeared_counter = Counter(np.unique(gt))

    no_edges_gt = gt[:, :, remove_px:-top_remove_px]
    appeared_no_edges_counter = Counter(np.unique(no_edges_gt))

    correct, correct_counter = is_completely_correct(prediction, gt)
    correct_no_edges, correct_no_edges_counter = is_completely_correct(
        prediction[:, :, remove_px:-top_remove_px], no_edges_gt
    )

    print(f"{correct=}, {correct_no_edges=}")
    # prediction = torch.nn.functional.one_hot(prediction, num_classes=50)[..., 1:]
    # gt = torch.nn.functional.one_hot(gt, num_classes=50)[..., 1:]

    # tp, fp, fn, tn = smp.metrics.get_stats(prediction, gt, mode="multilabel")
    tp, fp, fn, tn = smp.metrics.get_stats(prediction - 1, gt - 1, mode="multiclass", num_classes=50, ignore_index=-1)
    kwargs = dict(tp=tp, fp=fp, fn=fn, tn=tn, reduction="micro")
    stats = dict(
        # prediction=prediction,
        # gt=gt,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        correct=correct,
        correct_counter=correct_counter,
        correct_no_edges=correct_no_edges,
        correct_no_edges_counter=correct_no_edges_counter,
        appeared_counter=appeared_counter,
        appeared_no_edges_counter=appeared_no_edges_counter,
        iou=smp.metrics.iou_score(**kwargs),
        f1=smp.metrics.f1_score(**kwargs),
        f2=smp.metrics.fbeta_score(**kwargs, beta=2),
        acc=smp.metrics.accuracy(**kwargs),
        recall=smp.metrics.recall(**kwargs),
        precision=smp.metrics.precision(**kwargs),
    )
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            stats[key] = value.numpy()
    return stats


def single_plot(ax, palette, background, instance, labels, title, *, font_scale=1):
    # plt.subplot(1, len(plots), index + 1)

    # colors = plt.cm.prism(np.linspace(0, 1, 50))
    colors = np.array([list(c.floats()) for c in palette.values()])
    colors[0] = 0

    # if background is not None:
    #    colors[:, -1] = 0.1
    #    colors[0, -1] = 0
    if title and "correct" in title.lower():
        colors = [[0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1]]

    vmax = len(colors) - 1
    cmap = plt.cm.colors.ListedColormap(colors)

    ax.set_xticks([])
    ax.set_yticks([])

    if title is not None:
        ax.set_title(title, fontsize=8 * font_scale)

    if labels is not None:
        for i in np.unique(instance)[1:]:
            centroid = np.median(np.nonzero(instance == i), axis=1)
            label = labels.get(i, str(i))
            is_disc = "-" in label
            if not is_disc:
                ax.text(
                    centroid[1] + (45 if not is_disc else 0),
                    centroid[0],
                    label,
                    ha="left" if not is_disc else "center",
                    va="center",
                    fontsize=4 * font_scale,
                    color="black",
                    # path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")],
                )
                ax.plot([centroid[1], centroid[1] + 43], [centroid[0], centroid[0]], color="black", linewidth=0.3)

    if background is not None:
        ax.imshow(background)
    if instance is not None:
        ax.imshow(instance, cmap=cmap, vmin=0, vmax=vmax, interpolation="none")


def plot_npz(npz, save_to: str = None, show: bool = False, slices: range = None, save_separate=False, stats=None):
    get_transposed = lambda key: npz[key].T[::-1]

    if "gt_instances" in npz:
        gt_instances = npz["gt_instances"]
        gt = gt_instances.T[::-1]
    elif list(np.unique(npz["gt"])) == [0]:
        gt_instances = np.zeros_like(npz["segmentation"])
        gt = get_transposed("gt")
    else:
        gt_instances, _ = separate_rois_with_labels(npz["gt"])
        gt = get_transposed("gt")

    gt_instances = gt_instances.T[::-1]

    instances = get_transposed("instances")

    if "instances_post_processed" in npz:
        output = get_transposed("instances_post_processed")
    elif "segmentation" in npz:
        output = get_transposed("segmentation")
    else:
        output = instances

    if stats is None:
        stats = get_stats(npz["instances_post_processed"], npz["gt_instances"])
        # stats = get_stats(get_transposed("instances_post_processed").T, gt_instances.T)

    footer = (
        f"iou={stats['iou']:.3f}  "
        f"f1={stats['f1']:.3f}  "
        f"f2={stats['f2']:.3f}  "
        f"prec={stats['precision']:.3f}  "
        f"rec={stats['recall']:.3f}  "
        f"acc={stats['acc']:.3f}  "
    )

    original = get_transposed("original")[:, :, 0, :]

    if slices is None:
        slices = range(output.shape[2] // 2 - 1, output.shape[2] // 2)

    slices = range(slices.start, min(slices.stop, output.shape[2]), slices.step)

    # id_to_label = npz["id_to_label"].item()

    gt_id_to_label = id_to_label = dict(zip(range(1, 50), get_labels_for_n_classes(49)))

    # Binary dilation
    if use_highlighting := False:
        from scipy.ndimage import binary_dilation

        vertebra = output % 2 == 1
        discs = (output % 2 == 0) & (output != 0)
        highlight_vertebra = binary_dilation(vertebra) & ~vertebra
        highlight_discs = binary_dilation(discs) & ~discs
        highlight_overall = binary_dilation(output != 0) & ~(output != 0)

        highlight = highlight_vertebra | highlight_discs | highlight_overall
        highlighted = original + highlight * original.max() / 2

    alternating = get_alternating_palette()
    wkbs_palette = get_wk_bs_palette()

    plots = [
        [original, None, None, "MRI", alternating],
        # [highlighted, instances, id_to_label, "Orig+Out", alternating],
        # [original, output, None, "Seg", wkbs_palette],
        [None, instances, gt_id_to_label, "Output", alternating],
        # [None, instances, id_to_label, "Output", alternating],
        [None, gt_instances, gt_id_to_label, "GT", alternating],
    ]

    if "instances_post_processed" in npz:
        post_processed = get_transposed("instances_post_processed")
        plots.insert(-1, [None, post_processed, id_to_label, "Post-processed", alternating])

        correct = (post_processed == gt_instances).astype(int)
        correct[post_processed != gt_instances] = 2
        correct[gt_instances == 0] = 0
        title = "Incorrect"
        if stats["correct_no_edges"]:
            title = "Correct (NE)"
        if stats["correct"]:
            title = "Correct"

        incorrect_ids = [
            v for v, k in (stats["appeared_no_edges_counter"] - stats["correct_no_edges_counter"]).items() if k > 0
        ]
        title += f" ({', '.join(map(str, incorrect_ids))})"
        plots.append([None, correct, None, title, alternating])

    if "cropped_segmentation" in npz:
        cropped_segmentation = get_transposed("cropped_segmentation")
        plots.append([None, cropped_segmentation, None, "Segmentation", wkbs_palette])

    # if index == len(plots) // 2:
    thin_by = 30
    thinner = slice(thin_by, original.shape[1] - thin_by)

    # print(original)
    is_not_zero = np.abs(original) > 1e-6
    # print(np.any(is_not_zero, axis=(1, 2)).shape)
    # print(np.any(is_not_zero, axis=(1, 2)))
    zero_layers_at_start = np.argmax(np.any(is_not_zero, axis=(1, 2)))
    zero_layers_at_end = np.argmax(np.any(is_not_zero, axis=(1, 2))[::-1])
    # print(f"{zero_layers_at_start=}, {zero_layers_at_end=}")
    shorter = slice(zero_layers_at_start, original.shape[0] - zero_layers_at_end)

    save_to_ax = save_to.parent / "axes"
    save_to_ax.mkdir(exist_ok=True, parents=True)

    for slice_index in slices:
        fig, axes = plt.subplots(1, len(plots))

        for index, (ax, (background, instance, labels, title, palette)) in enumerate(zip(axes, plots)):
            if background is not None:
                background = background[shorter, thinner, slice_index]
            if instance is not None:
                instance = instance[shorter, thinner, slice_index]
            single_plot(ax, palette, background, instance, labels, title)

            if save_separate:
                single_fig, single_ax = plt.subplots(1, 1)
                single_plot(single_ax, palette, background, instance, labels, title)
                single_fig.savefig(
                    save_to_ax / f"slice{slice_index}_{title.replace(' ', '')}.png", dpi=600, transparent=True
                )
                single_fig.tight_layout(pad=0)
                plt.close(single_fig)

        # text_kwargs = dict(ha="center", va="center", fontdict={"family": "monospace"}, color="black", fontsize=6)
        # for at in np.linspace(0.05, 0.3, 4):
        #     fig.text(0.5, at, footer, **text_kwargs)

        fig.tight_layout(pad=0)

        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_to.with_name(f"{save_to.stem}_slice{slice_index:02}{save_to.suffix}"), dpi=600)
        if show:
            plt.show()
        plt.close()

import numpy as np
import vispy.scene
from matplotlib import pyplot as plt
from pandas import DataFrame
from vispy.scene import visuals

from spine_segmentation.annotators.vector_annotator import VectorAnnotator
from spine_segmentation.resources.paths import PLOTS_PATH
from spine_segmentation.resources.preloaded import get_measure_statistics


def _evaluate_and_save_off_by(patient_centers):
    off_by_table = np.ones((patient_centers.shape[0], patient_centers.shape[1])) * (-1)
    annotator = VectorAnnotator()
    for i in range(patient_centers.shape[0]):
        off_by = annotator.evaluate_each_roi_on_patient(6, i)
        for j in sorted(off_by):
            off_by_table[i, j] = off_by[j]
        if i % 10 == 0:
            print(f"Saving at {i}/{patient_centers.shape[0]}")
            np.savez_compressed("classified_as_table.npz", classified_as_table=off_by_table)

    # print confusion matrix
    off_by_table = np.nan_to_num(off_by_table)
    off_by_table = off_by_table.astype(int)
    print("Confusion matrix:")
    print(DataFrame(off_by_table).value_counts().to_string())


def _create_confusion_matrix():
    from sklearn.metrics import confusion_matrix

    data = np.load("classified_as_table.npz")["classified_as_table"]

    data = data[:2901, :40]

    # Assuming you have a numpy array called 'data' with shape (2900, 47)
    classification_guesses = data  # Last column represents the classification guesses
    actual_indices = np.arange(data.shape[1])[None].repeat(
        data.shape[0], axis=0
    )  # Array of indices from 0 to (number of data points - 1)

    # Generate confusion matrix
    cm = confusion_matrix(actual_indices.flatten(), classification_guesses.flatten())

    # Plot confusion matrix
    for i, matrix in enumerate([cm, np.log(cm)]):
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(f"Confusion matrix {'(logarithmic)' if i == 1 else ''}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(PLOTS_PATH / f"confusion_matrix_{i}.png")
        plt.show()

    # Print the confusion matrix
    print(cm)


def _plot_tsne_subset_evaluate(patient_centers, subset_size):
    patient_centers = patient_centers[:subset_size]
    patient_centers = np.nan_to_num(patient_centers)

    tsne_results = np.load("tsne_results.npy")
    off_by_table = np.load("off_by_table.npz")["off_by_table"]

    face_colors = []
    for i in range(subset_size):
        face_colors.extend(
            [
                {
                    -1: (0, 0, 0, 0.1),
                    0: (0, 1, 0, 0.8),
                    2: (1, 0.5, 0, 0.8),
                }.get(off_by_table[i, j], (1, 0, 0, 0.9))
                for j in range(off_by_table.shape[1])
            ]
        )
    DataFrame(tsne_results).plot.scatter(x=0, y=1, c=face_colors, s=0.3)
    plt.title(
        f"t-SNE of {int(tsne_results.shape[0] / patient_centers.shape[1])} patients, {tsne_results.shape[0]} ROIs"
    )
    labels = ["incorrect", "off by 2", "off by 0", "unknown"]
    colors = ["red", "orange", "green", "grey"]
    handles = [plt.Line2D([], [], color=color, marker="o", linestyle="None") for color in colors]
    plt.legend(handles, labels)
    plt.savefig("tsne.png", dpi=300)
    plt.show()


def _flatten_to_coordinates(separate_roi_xyz):
    """Input shape: (10k, 51, 3), output shape: (510k, 3)"""
    return separate_roi_xyz.reshape(-1, 3)


def plot_3d_subset_evaluate(patient_centers, subset_size):
    # annotator = VectorAnnotator()
    patient_centers = patient_centers[:subset_size]
    face_colors = []
    off_by_table = np.load("off_by_table.npz")["off_by_table"]
    print(off_by_table.shape)
    for i in range(off_by_table.shape[0]):
        # off_by = annotator.evaluate_each_roi_on_patient(6, i)
        # off_by = off_by_table[i]
        face_colors.extend(
            [
                {
                    -1: (1, 1, 1, 0.2),
                    0: (0, 1, 0, 0.8),
                    2: (1, 0.5, 0, 0.8),
                }.get(off_by_table[i, j], (1, 0, 0, 0.9))
                for j in range(off_by_table.shape[1])
            ]
        )
        # roi_count = patient_centers.shape[1]
        # face_colors.extend([(1, 1, 1, 0.2)] * (roi_count - len(face_colors) % roi_count))
    plot_3d(patient_centers, face_colors)


def plot_3d_with_basic_colors(patient_centers, coloring_type="alternating", vectors=None, separate=False):
    face_colors = []

    if coloring_type == "alternating":
        colors = [(1, 0, 0, 0.7) if i % 2 else (1, 1, 1, 0.7) for i in range(patient_centers.shape[1])]
        face_colors = colors * patient_centers.shape[0]

    if coloring_type == "unique_hex":
        for i in range(patient_centers.shape[0]):
            col = [1, (i // 255) / 255, (i % 255) / 255, 1]
            face_colors.extend([tuple(col)] * patient_centers.shape[1])

    if coloring_type == "single_by_hex":
        real_i = None
        for i in range(patient_centers.shape[0]):
            col = [1, (i // 255) / 255, (i % 255) / 255, 1]
            hex = 0xFF0D64
            is_same = hex / 255 == col[0] * 2**16 + col[1] * 2**8 + col[2]
            col[-1] = 1 if is_same else 0.0
            face_colors.extend([tuple(col)] * patient_centers.shape[1])
            if col[-1] == 1:
                real_i = i
        patient_centers = patient_centers[real_i : real_i + 1]
        face_colors = face_colors[real_i * patient_centers.shape[1] : (real_i + 1) * patient_centers.shape[1]]

    plot_3d(patient_centers, face_colors, vectors, separate=separate)


def plot_3d(patient_centers, face_colors, patient_vectors_list=None, separate=False):
    centers = _flatten_to_coordinates(patient_centers)
    """Centers shape: (510k, 3)"""

    if separate:
        separate_by = np.arange(patient_centers.shape[0]) * 50
        patient_centers[..., 1] += separate_by[:, np.newaxis]

    canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
    view = canvas.central_widget.add_view()
    # ArrowVisual()

    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(centers, edge_width=1, face_color=face_colors, size=2 + separate * 5)
    view.add(scatter)

    v1 = _flatten_to_coordinates(patient_vectors_list[0])
    v2 = _flatten_to_coordinates(patient_vectors_list[1])
    diff = np.sum((v1 - v2) ** 2, axis=1)[..., np.newaxis]

    alpha = 1 if separate else 0.5
    vector_colors = [(1, 0, 0, alpha), (1, 1, 1, alpha), (1, 1, 0, alpha)]

    if patient_vectors_list is not None:
        for arrow_colors, patient_vectors in zip(vector_colors, patient_vectors_list):
            vectors = _flatten_to_coordinates(patient_vectors)
            # Merge shape (500k, 3) with (500k, 3) to (500k, 2, 3)
            # arrows = np.concatenate([centers, centers+vectors*(1+separate*9)], axis=1).reshape(-1, 2, 3)
            arrows = np.concatenate([centers, centers + vectors * (diff * 100 + 3 * separate)], axis=1).reshape(
                -1, 2, 3
            )

            # (500k, 2, 3) to (500k, 6)
            arrows_dir = arrows.reshape(-1, 6)

            # arrow_colors = np.repeat(face_colors, 2, axis=0)
            arrow_visuals = visuals.Arrow(
                arrows,
                width=2,
                color=arrow_colors,
                arrow_size=1,
                connect="segments",
                arrow_type="stealth",
                # arrows=arrows_dir,
                arrow_color=arrow_colors,
            )
            view.add(arrow_visuals)

    view.camera = "turntable"  # or try 'arcball'

    # Camera centered at 00
    view.camera.center = (0, 0, 0)

    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)
    vispy.app.run()


def main():
    stats = get_measure_statistics()
    # _create_confusion_matrix()
    plot_3d_subset_evaluate(stats.centers, stats.all_vectors)
    # _evaluate_and_save_off_by(stats.centers)
    # plot_3d_with_basic_colors(stats.centers, "alternating", stats.all_vectors, separate=True)


if __name__ == "__main__":
    main()

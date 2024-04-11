from pathlib import Path

from spine_segmentation.datasets.sample import SampleIterator
from spine_segmentation.inference.onnx_model import ONNXInferenceModel
from spine_segmentation.plotting.plot_slice import plot_npz
from spine_segmentation.utils.log_dir import get_next_log_dir

prefix = Path("define path")

edge_case_paths = [
    "1b7cd7a47cc95032792b48fd5b6f5138.npz",
    "3f776bfd85beb75318e5e2f51dcc1cef.npz",
    "68958c058fc83017a3d8e5e00a9321fb.npz",
    "8897172fff245644f0d3acf7552f090d.npz",
    "00cccb4bb3680540355ea2dea7c7a936.npz",
    "0eba72dd557eb2397e3b3cde07166cb3.npz",
    "b4d9152a99be88e7fbfde19547b42e3e.npz",
    "0ee5499747f9332f9fa4036eae43e2f3.npz",
    "9080e06c4c8601213098acaccee44e46.npz",
    "5781cb28b20d9fc416aec7b04b842e07.npz",
    "5530f3e564d5ae1ec2508dce02fef34f.npz",
    "74e0c90952a4a3022b188a1b0ed28ceb.npz",
    "6ede6800b5ab8915fea0ffde354a662b.npz",
    "b2ec9888dce9c6eb145cd3f3666ffda7.npz",
    "ade5f227c3da97d63fcb40ddd2fc3e30.npz",
    "22bf344080942957ae839dc344377658.npz",
    "c0fd29f1b15373ecd5edd2d5c5817855.npz",
    "76e1ab95dab4d0ec002211a1bfbb458b.npz",
    "d329cc194c630ced2401ec1baafee0fc.npz",
    "a509f7e5ca2595ff20351689069209ee.npz",
    "40a706777ec5274ed83ffa73d65e0ff8.npz",
    "7b409419cced71aa135f429afb908db0.npz",
    "a461064d250933289c43f37ddfa28696.npz",
    "85ddb7690a26ba0e416fa315c65b14f8.npz",
    "4644386d37e5d829f7ea2e8d5f24f5b0.npz",
    "c8248089d8ae38bfa8f62d215afffc8b.npz",
    "038e045c136ff888060da7ec3bacc301.npz",
    "481794b8cc08d88cdaeb076998cf39b1.npz",
    "7fe1fa781c990aaf70e9e9ee2ebbd911.npz",
    "5f3a78ed8e015e8422503b29f15519e8.npz",
    "778dbc632584c4039576b64ee9812553.npz",
    "3280b390a96b2616b71e9db3ed828b53.npz",
    "6f18fb717f873c42b326ad8fac7c971f.npz",
    "3d8f8d80b0d95d5ba432866856ab6c70.npz",
    "869d8379b5ddd582043f6fa845b878a7.npz",
]


def main():
    onnx = ONNXInferenceModel.get_best_segmentation_model()
    onnx.load_index_list()
    npz_path_to_zip_path = {npz_path: onnx.index_to_zip_path[i] for i, npz_path in onnx.index_to_npz_path.items()}
    zip_paths = [npz_path_to_zip_path[prefix / path] for path in edge_case_paths]
    sample_iterator = SampleIterator(zip_paths, add_adjacent_slices=True)

    log_dir = get_next_log_dir()

    for image, gt, sample, path in sample_iterator:
        print(image.shape, gt.shape)
        npz = onnx.inference(image, gt=gt)
        save_to = log_dir / "edge_cases" / f"{sample.dicom.PatientID}.png"
        plot_npz(npz, save_to, slices=range(0, 100))


if __name__ == "__main__":
    main()

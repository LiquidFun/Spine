from ydata_profiling import ProfileReport

from spine_segmentation.resources.preloaded import get_measure_statistics

if __name__ == "__main__":
    profile_report = ProfileReport(get_measure_statistics().dicom_metadata)
    profile_report.to_file("profiling.html")

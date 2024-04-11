from spine_segmentation.resources.preloaded import get_measure_statistics


def stat(df):
    import matplotlib.pyplot as plt

    # Calculate the BMI using the DICOM;PatientSize (meters) and DICOM;PatientWeight (kg) columns
    df["BMI"] = df["DICOM;PatientWeight"] / (df["DICOM;PatientSize"] ** 2)

    # Extract the Volume data from the C2-3;DISC_VOLUME column
    volume = df["T10-11;DISC_VOLUME"]
    # df.to_html("index.html")
    # print(df.corr())

    # Plot the BMI over Volume
    plt.scatter(df["BMI"], volume, s=1)
    # plt.ylabel('Volume (mm^3)')
    plt.ylabel("Volume (mm^3)")
    plt.xlabel("BMI")
    plt.title("Volume over BMI")
    plt.show()


stat(get_measure_statistics().raw_statistics)

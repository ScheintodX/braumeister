import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import DataFrame

TEMPERATURE_FIX = 1.113
ERROR_SCALE = 1000.0

def load_data( timebase:str ) -> DataFrame:

    def load_and_fix(filename:str, valuename:str) -> DataFrame:
        theframe = pd.read_csv(f"data/{timebase}_{filename}").iloc[:, 1:]
        theframe["time"] = pd.to_datetime(theframe["time"])
        theframe.set_index("time", inplace=True)
        theframe_resampled = theframe.resample("1s").mean()
        theframe_resampled = theframe_resampled.rename(columns={"value": valuename} )
        return theframe_resampled

    # Load datasets
    tmprset = load_and_fix("temp.csv", "temperature")
    pwerset = load_and_fix("powerfactor.csv", "powerfactor")
    systset = load_and_fix("system.csv", "system")
    dmprset = load_and_fix("damper.csv", "damper")

    # Step 1: Resample to 1-second intervals
    # Step 2: Interpolate gaps in the 1-second data
    # Step 3: Downsample to 30-second intervals
    tmprset_resampled = tmprset.resample("1s").mean()
    tmprset_resampled = tmprset_resampled.interpolate(method="time")
    tmprset_30s = tmprset_resampled.resample("30s").mean()

    pwerset_resampled = pwerset.resample("1s").ffill()
    pwerset_30s = pwerset_resampled.resample("30s").last()

    systset_resampled = systset.resample("1s").ffill()
    systset_30s = systset_resampled.resample("30s").last()

    dmprset_resampled = dmprset.resample("1s").ffill()
    dmprset_30s = dmprset_resampled.resample("30s").last()

    # Combine into one
    final_dataset = pd.concat([tmprset_30s, pwerset_30s, systset_30s, dmprset_30s], axis=1)

    # Fix powerfactor by system state
    final_dataset["powerfactor"] = final_dataset["powerfactor"].where(final_dataset["system"] != 0, 0)

    # Fix temp sensor
    final_dataset["temperature"] = final_dataset["temperature"] * TEMPERATURE_FIX

    return final_dataset


def plot_data( final_dataset : DataFrame ):

    plt.close('all')

    # Create a figure and axis for the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Temperature on the first y-axis
    ax1.plot(final_dataset.index, final_dataset["temperature"], label="Temperature (°C)", color="tab:red")
    ax1.set_xlabel("time")
    ax1.set_ylabel("Temperature (°C)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Create a second y-axis for Powerfactor
    ax2 = ax1.twinx()
    ax2.plot(final_dataset.index, final_dataset["powerfactor"], label="Powerfactor", color="tab:blue")
    ax2.set_ylabel("Powerfactor (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax3 = ax1.twinx()
    ax3.plot(final_dataset.index, final_dataset["damper"], label="Damper", color="tab:green")
    ax3.set_ylabel("Damper", color="tab:green")
    ax3.tick_params(axis="y", labelcolor="tab:green")

    if "sim" in final_dataset.columns:
        ax1.plot(final_dataset.index, final_dataset["sim"], label="Simulated Temp (°C)", color="orange")

    if "error" in final_dataset.columns:
        ax1.plot(final_dataset.index, final_dataset["error"], label=f"Temp Error (°C)", linestyle="--", color="orange")

    # Format x-axis to show only hour:min:sec
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # Add grid and title
    plt.title("Temperature McTemperatureface")
    ax1.grid(True)

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.85))

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = load_data()
    plot_data(dataset)

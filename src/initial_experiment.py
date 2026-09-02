import numpy as np
import pandas as pd


def main():
    # Synthetic data for initial environment and workflow validation
    sample_data = pd.DataFrame(
        {
            "day": ["Day 1", "Day 2", "Day 3"],
            "hotspot_count": [5, 8, 3],
        }
    )

    print("Initial Experiment - Kalimantan Hotspot Forecasting")
    print()
    print(sample_data)
    print()
    print(f"Average hotspot count: {np.mean(sample_data['hotspot_count']):.2f}")
    print("Initial experiment completed successfully.")


if __name__ == "__main__":
    main()
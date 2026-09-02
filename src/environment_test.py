import sys

import numpy as np
import pandas as pd
import requests
import sklearn


def main():
    print("Kalimantan Hotspot Forecasting")
    print("Development environment is ready.")
    print()
    print(f"Python       : {sys.version.split()[0]}")
    print(f"Pandas       : {pd.__version__}")
    print(f"NumPy        : {np.__version__}")
    print(f"Scikit-learn : {sklearn.__version__}")
    print(f"Requests     : {requests.__version__}")


if __name__ == "__main__":
    main()
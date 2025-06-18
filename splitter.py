import pandas as pd
from sklearn.model_selection import train_test_split

def splitter(path: str):

    """
    Loads data from a CSV file and splits it into training and testing sets.

    Parameters:
    -----------
    path : str
        Path to the CSV file.

    Returns:
    --------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        Split training and testing features and target.
    """

    df = pd.read_csv(path)

    if 'price' not in df.columns:
        raise ValueError(f"Target column 'price' not found in dataset.")

    X = df.drop(columns=['price'])
    y = df['price']

    return train_test_split(X, y, test_size=0.2, random_state=42)
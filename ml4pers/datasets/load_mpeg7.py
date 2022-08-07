import pkg_resources
import pandas as pd


def load_mpeg7():
    """Load the persistence diagrams from the MPEG7 dataset

    """
    
    stream = pkg_resources.resource_stream(__name__, 'data/mpeg7.pickle')
    return pd.read_pickle(stream)
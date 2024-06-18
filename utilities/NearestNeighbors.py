import pandas as pd
from scipy.spatial import KDTree
import numpy as np


class NearestTreeNeighbors:
    def __init__(self, tree_df: pd.DataFrame, radius: float = 0.0001):
        """
        @TODO
        """
        self.df = tree_df
        self.kd = KDTree(tree_df[["longitude", "latitude"]].values)
        self.radius = radius

    def query_neighbors(self, longitude: float, latitude: float) -> pd.DataFrame:
        """
        @TODO
        """
        nearest_tree_idx = self.kd.query_ball_point(
            [longitude, latitude], self.radius, p=2.0
        )

        neighbors = self.df.loc[nearest_tree_idx]
        # Prevent returning the same tree
        neighbors = neighbors[
            (neighbors["longitude"] != longitude)
            & (neighbors["latitude"] != latitude)
        ]

        neighbors["distance_sq"] = np.square(
            neighbors["longitude"] - longitude
        ) + np.square(neighbors["latitude"] - latitude)
        neighbors = neighbors.sort_values(by=["distance_sq"], ascending=True)

        return neighbors

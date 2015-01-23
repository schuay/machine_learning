import dataset as ds

class RegressionInstanceI(ds.InstanceI):
    def x(self):
        """Returns a list of features used as prediction inputs."""
        raise NotImplementedError("Please implement this yourself.")

    def y(self):
        """Returns a list of features used as prediction outputs."""
        raise NotImplementedError("Please implement this yourself.")

class RegressionDatasetI(ds.DatasetI):
    pass

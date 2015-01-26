class DatasetI:
    def data(self):
        raise NotImplementedError("Please implement this yourself.")

    def target(self):
        raise NotImplementedError("Please implement this yourself.")

    def name(self):
        """Returns the name of this dataset."""
        raise NotImplementedError("Please implement this yourself.")

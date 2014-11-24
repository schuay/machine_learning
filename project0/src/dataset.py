class InstanceI:
    def features(self):
        """Returns a dictionary mapping features to values."""
        raise NotImplementedError("Please implement this yourself.")

class DatasetI:
    def instances(self):
        """Returns a list of all instances in this dataset."""
        raise NotImplementedError("Please implement this yourself.")

    def name(self):
        """Returns the name of this dataset."""
        raise NotImplementedError("Please implement this yourself.")

    def kind(self):
        """Returns the kind of this dataset."""
        raise NotImplementedError("Please implement this yourself.")

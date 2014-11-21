class ClassificationInstanceI:
    def instance_class(self):
        """Returns the class of this instance."""
        raise NotImplementedError("Please implement this yourself.")

    def features(self):
        """Returns a dictionary mapping features to values."""
        raise NotImplementedError("Please implement this yourself.")

class ClassificationDatasetI:
    def classes(self):
        """Returns a list of all classes in this dataset."""
        raise NotImplementedError("Please implement this yourself.")

    def instances(self):
        """Returns a list of all instances in this dataset."""
        raise NotImplementedError("Please implement this yourself.")

    def name(self):
        """Returns the name of this dataset."""
        raise NotImplementedError("Please implement this yourself.")

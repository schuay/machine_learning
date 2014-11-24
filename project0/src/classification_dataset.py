import dataset as ds

class ClassificationInstanceI(ds.InstanceI):
    def instance_class(self):
        """Returns the class of this instance."""
        raise NotImplementedError("Please implement this yourself.")

class ClassificationDatasetI(ds.DatasetI):
    def classes(self):
        """Returns a list of all classes in this dataset."""
        raise NotImplementedError("Please implement this yourself.")

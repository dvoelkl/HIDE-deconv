"""
=====================================================
Methods for accessing and storing configurations
necessary for the command line interface
=====================================================
"""

import json


class hidedeconv_config:
    """
    Configuration of HIDE command line interface
    """

    def __init__(self) -> None:

        # Configuration for training data
        self.sc_file_name = ""
        self.sub_ct_col = ""
        self.higher_ct_cols = []

        self.bulk_file_name = ""

        self.n_genes = -1
        self.n_train_bulks = -1
        self.n_cells_per_bulk = -1
        self.n_hide_iter = -1

        self.preprocessed = False
        self.domainTransfer = True
        self.trained = False

    def to_dict(self) -> dict[str, type]:
        """
        Get representation of configuration as dictionary.

        Returns
        -------
        dict[str, type]
            Dictionary with all attributes of configuration class.

        """
        d = {
            "sc_file_name": self.sc_file_name,
            "sub_ct_col": self.sub_ct_col,
            "higher_ct_cols": self.higher_ct_cols,
            "bulk_file_name": self.bulk_file_name,
            "n_genes": self.n_genes,
            "n_train_bulks": self.n_train_bulks,
            "n_cells_per_bulk": self.n_cells_per_bulk,
            "n_hide_iter": self.n_hide_iter,
            "preprocessed": self.preprocessed,
            "domainTransfer": self.domainTransfer,
            "trained": self.trained,
        }

        return d

    @staticmethod
    def from_dict(d: dict[str, type]) -> "hidedeconv_config":
        """
        Create an instance of hidedeconv_config from a dictionary.

        Parameters
        ----------
        d : dict[str, type]
            Dictionary containing all attributes of the hidedeconv_config class.

        Returns
        -------
        hidedeconv_config
            Instance of hidedeconv_config filled with the values in the dictionary.

        """
        hconf = hidedeconv_config()

        hconf.sc_file_name = d["sc_file_name"]
        hconf.sub_ct_col = d["sub_ct_col"]
        hconf.higher_ct_cols = d["higher_ct_cols"]
        hconf.bulk_file_name = d["bulk_file_name"]
        hconf.n_genes = int(d["n_genes"])
        hconf.n_train_bulks = int(d["n_train_bulks"])
        hconf.n_cells_per_bulk = int(d["n_cells_per_bulk"])
        hconf.n_hide_iter = int(d["n_hide_iter"])
        hconf.preprocessed = d["preprocessed"]
        hconf.domainTransfer = d["domainTransfer"]
        hconf.trained = d["trained"]

        return hconf

    def save(self, path: str) -> None:
        d = self.to_dict()
        conf_json = json.dumps(d)

        with open(path, "w") as json_file:
            json_file.write(conf_json)

    @staticmethod
    def load(path: str) -> "hidedeconv_config":

        with open(path, "r") as json_file:
            conf_json = json_file.read()
            d = json.loads(conf_json)

        return hidedeconv_config.from_dict(d)

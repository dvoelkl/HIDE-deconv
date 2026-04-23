"""
=====================================================
Utility functions for samples sheets
=====================================================
"""


def sample_ids_valid(sample_ids: list[str], bulk_names: list[str]) -> bool:
    """
    Checks if the sample ids from a sample sheet are compatible with the bulk names.

    Returns
    -------
    bool
        True, if bulk_names are subset of sample_ids.
    """

    return set(bulk_names).issubset(set(sample_ids))

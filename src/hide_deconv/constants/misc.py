"""
=====================================================
Miscellaneous constants
=====================================================
"""

MODEL_HIDE = 0

CELLXGENE_DATASETS_URL = "https://api.cellxgene.cziscience.com/curation/v1/datasets"
CELLXGENE_COLLECTIONS_URL = "https://cellxgene.cziscience.com/collections"
CELLXGENE_PLATFORM_CITATION = (
    "CZI Cell Science Program, Abdulla S, Aevermann B, et al. CZ CELLxGENE Discover: "
    "a single-cell data platform for scalable exploration, analysis and modeling of "
    "aggregated data. Nucleic Acids Res. 2025;53(D1):D886-D900. "
    "https://doi.org/10.1093/nar/gkae1142"
)
CELLXGENE_API_TIMEOUT = (10, 300)  # (connect, read) timeout in seconds
DOWNLOAD_TIMEOUT = (10, 120)  # (connect, read) timeout in seconds
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

DOMAIN_TRANSFER_BULK_COUNT = 1000
DOMAIN_TRANSFER_PREDS_PER_BULK = 5
DOMAIN_TRANSFER_ALPHA_WINDOW = 25
DOMAIN_TRANSFER_SEED = 2304

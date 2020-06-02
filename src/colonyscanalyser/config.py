"""
Package configuration
"""

# Images
DOTS_PER_INCH = 300
SUPPORTED_FORMATS = ["tif", "tiff", "png", "bmp"]

# Plates
PLATE_EDGE_CUT = 5
PLATE_LATTICE = (3, 2)
PLATE_SIZE = 90

# Colonies
COLONY_FIRST_AREA_MAX = 10
COLONY_DISTANCE_MAX = 2
COLONY_TIMEPOINTS_MIN = 5
COLONY_TIMESTAMP_DIFF_MAX = 20
COLONY_GROWTH_FACTOR_MIN = 4

# Output
CACHED_DATA_FILE_NAME = "cached_data"
DATA_DIR = "data"
OUTPUT_VERBOSE = 1
OUTPUT_PLOTS = 1
PLOTS_DIR = "data"

# Runtime
MULTIPROCESSING = True
USE_CACHED_DATA = False
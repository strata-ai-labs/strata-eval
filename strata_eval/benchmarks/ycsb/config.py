"""Default parameters for YCSB workloads."""

# Number of records to pre-load into the database.
DEFAULT_RECORD_COUNT = 100_000

# Number of operations to execute in the run phase.
DEFAULT_OPERATION_COUNT = 100_000

# Number of fields per record (YCSB standard: 10).
DEFAULT_FIELD_COUNT = 10

# Length of each field value in bytes (YCSB standard: 100).
DEFAULT_FIELD_LENGTH = 100

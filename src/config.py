"""Central configuration so every script uses the same paths, dates and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # put the downloaded daily files here
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
TZ = "Europe/Rome"                          # raw timestamps are UTC epoch milliseconds
START = "2013-11-01 00:00"                  # local time
N_DAYS = 62                                 # Nov 1 -> Jan 1 inclusive
SLOT_MINUTES = 10
N_SLOTS = N_DAYS * 24 * 60 // SLOT_MINUTES  # 8928
N_SQUARES = 10_000                          # square ids 1..10000

SPECIAL_SQUARES = [4159, 4556]

# Chronological split (local time)
TRAIN_END = "2013-12-08 23:50"
VAL_START, VAL_END = "2013-12-09 00:00", "2013-12-15 23:50"
TEST_START, TEST_END = "2013-12-16 00:00", "2013-12-22 23:50"

RAW_COLUMNS = ["square_id", "time_interval", "country_code",
               "sms_in", "sms_out", "call_in", "call_out", "internet"]

# Forecasting setup
WEEK = 1008                                  # 10-min slots per week
MAX_WINDOW = 432                             # longest NN input window allowed (3 days)
# All neural-network configurations use the SAME training targets, so window /
# feature experiments are compared on identical samples: first target = Nov 11 00:00.
FIRST_NN_TARGET = WEEK + MAX_WINDOW          # = 1440
HOLIDAYS = ["2013-11-01", "2013-12-07", "2013-12-08", "2013-12-25", "2013-12-26", "2014-01-01"]
N_TIMING_REPEATS = 5

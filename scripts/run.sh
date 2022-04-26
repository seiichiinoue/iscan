# global params
NUM_ITERATION=2000
KAPPA_PHI=4.0
KAPPA_PSI=100.0
KAPPA_PHI_START=100
KAPPA_PHI_INTERVAL=50
MIN_SNIPPET_LENGTH=1
MIN_WORD_COUNT=5
DAY=$(date "+%m%d")

# for japanese corpus settings (year)
# START_YEAR=1860
# END_YEAR=2000
# YEAR_INTERVAL=40

# for without CLMET corpus
START_YEAR=1810
END_YEAR=2010
YEAR_INTERVAL=20

# local params
TARGET_WORD=$1
# INPUT_DATA=./data/ja_v5/${TARGET_WORD}.txt
INPUT_DATA=./data/en_v3/${TARGET_WORD}.txt
BINARY_PATH=./results/bin/${TARGET_WORD}_${DAY}.model
LOG_PATH=./results/log/${TARGET_WORD}_${DAY}

./scan -data_path=$INPUT_DATA \
       -save_path=$BINARY_PATH \
       -kappa_phi=$KAPPA_PHI \
       -kappa_psi=$KAPPA_PSI \
       -kappa_phi_start=$KAPPA_PHI_START \
       -kappa_phi_interval=$KAPPA_PHI_INTERVAL \
       -min_snippet_length=$MIN_SNIPPET_LENGTH \
       -min_word_count=$MIN_WORD_COUNT \
       -start_year=$START_YEAR \
       -end_year=$END_YEAR \
       -year_interval=$YEAR_INTERVAL \
       -num_iteration=$NUM_ITERATION > $LOG_PATH

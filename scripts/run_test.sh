# parameters
# sampler
NUM_TIMES=10
NUM_SENSES=5
CONTEXT_WINDOW_SIZE=10
VOCAB_SIZE_PER_SENSE=1000
RATIO_COMMON_VOCAB=0.0
NUM_SAMPLE_PER_TIME=500
SHIFT_TYPE="random"
WORD_PRIOR_TYPE="zipf"

# sb-scan
NUM_ITERATION=1000
SCALING_COEFF=1.0
KAPPA_PSI=100.0
MIN_WORD_COUNT=10

DAY=$(date "+%m%d")
INPUT_DATA=./data/pseudo/pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_vocab${VOCAB_SIZE_PER_SENSE}_common${RATIO_COMMON_VOCAB}_window${CONTEXT_WINDOW_SIZE}_sample${NUM_SAMPLE_PER_TIME}.txt
SUFFIX=pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_sample${NUM_SAMPLE_PER_TIME}_vocab${VOCAB_SIZE_PER_SENSE}_kappa_psi${KAPPA_PSI}_min_word${MIN_WORD_COUNT}_${DAY}
BINARY_PATH=./results/bin/${SUFFIX}.model
LOG_PATH=./results/log/${SUFFIX}

# generate pseudo data
python3 tools/create_pseudo_data.py --num-times $NUM_TIMES \
                                    --num-senses $NUM_SENSES \
                                    --context-window-size $CONTEXT_WINDOW_SIZE \
                                    --vocab-size-per-sense $VOCAB_SIZE_PER_SENSE \
                                    --ratio-common-vocab $RATIO_COMMON_VOCAB \
                                    --num-sample $NUM_SAMPLE_PER_TIME \
                                    --shift-type $SHIFT_TYPE \
                                    --word-prior-type $WORD_PRIOR_TYPE \
                                    --output-path $INPUT_DATA
# execute test
./scan -data_path=$INPUT_DATA \
       -save_path=$BINARY_PATH \
       -context_window_width=$CONTEXT_WINDOW_SIZE \
       -scaling_coeff=$SCALING_COEFF \
       -kappa_psi=$KAPPA_PSI \
       -min_word_count=$MIN_WORD_COUNT \
       -start_year=0 \
       -end_year=$NUM_TIMES \
       -year_interval=1 \
       -num_iteration=$NUM_ITERATION > $LOG_PATH

# output probabilities
./prob -model_path=$BINARY_PATH -use_npmi=false > results/out/${SUFFIX}

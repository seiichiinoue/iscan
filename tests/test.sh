# parameters
# sampler
NUM_TIMES=16
NUM_SENSES=2
CONTEXT_WINDOW_SIZE=10
VOCAB_SIZE_PER_SENSE=100
RATIO_COMMON_VOCAB=0.2
NUM_SAMPLE=1000
SHIFT_TYPE="s-curve"

# sb-scan
NUM_ITERATION=5000
SCALING_COEFF=1.0
SIGMA_COEFF=0.1

DAY=1013
INPUT_DATA=./tests/sampled/pseudo_sense${NUM_SENSES}_vocab${VOCAB_SIZE_PER_SENSE}_window${CONTEXT_WINDOW_SIZE}_sample${NUM_SAMPLE}.txt
BINARY_PATH=./bin/pseudo_$DAY.model
LOG_PATH=./log/out_pseudo_$DAY

# generate pseudo data
python3 tests/generate_sample.py --num-times $NUM_TIMES \
                           --num-senses $NUM_SENSES \
                           --context-window-size $CONTEXT_WINDOW_SIZE \
                           --vocab-size-per-sense $VOCAB_SIZE_PER_SENSE \
                           --ratio-common-vocab $RATIO_COMMON_VOCAB \
                           --num-sample $NUM_SAMPLE \
                           --shift-type $SHIFT_TYPE \
                           --output-path $INPUT_DATA
# execute test
./scan -data_path=$INPUT_DATA \
       -save_path=$BINARY_PATH \
       -context_window_width=$CONTEXT_WINDOW_SIZE \
       -scaling_coeff=$SCALING_COEFF \
       -sigma_coeff=$SIGMA_COEFF \
       -start_year=0 \
       -end_year=$NUM_TIMES \
       -year_interval=1 \
       -num_iteration=$NUM_ITERATION > $LOG_PATH

# output probabilities
./prob -model_path=$BINARY_PATH -use_npmi=false > scripts/out/pseudo_$DAY
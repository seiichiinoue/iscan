# parameters
# sampler
NUM_TIMES=16
NUM_SENSES=$1
CONTEXT_WINDOW_SIZE=10
VOCAB_SIZE_PER_SENSE=1000
RATIO_COMMON_VOCAB=0.0
NUM_SAMPLE=1000
SHIFT_TYPE="random"
WORD_PRIOR_TYPE="dirichlet"

# sb-scan
NUM_ITERATION=2000
SCALING_COEFF=1.0
SIGMA_COEFF=0.05
KAPPA_PSI=100.0
TOP_N_WORD=3000

DAY=$(date "+%m%d")
INPUT_DATA=./tests/sampled/pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_vocab${VOCAB_SIZE_PER_SENSE}_common${RATIO_COMMON_VOCAB}_window${CONTEXT_WINDOW_SIZE}_sample${NUM_SAMPLE}.txt
SUFFIX=pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_sample${NUM_SAMPLE}_vocab${VOCAB_SIZE_PER_SENSE}_kappa_psi${KAPPA_PSI}_top_n${TOP_N_WORD}_${DAY}
BINARY_PATH=./bin/${SUFFIX}.model
LOG_PATH=./log/out_${SUFFIX}

# generate pseudo data
python3 tests/sample_data.py --num-times $NUM_TIMES \
                             --num-senses $NUM_SENSES \
                             --context-window-size $CONTEXT_WINDOW_SIZE \
                             --vocab-size-per-sense $VOCAB_SIZE_PER_SENSE \
                             --ratio-common-vocab $RATIO_COMMON_VOCAB \
                             --num-sample $NUM_SAMPLE \
                             --shift-type $SHIFT_TYPE \
                             --word-prior-type $WORD_PRIOR_TYPE \
                             --output-path $INPUT_DATA
# execute test
./scan -data_path=$INPUT_DATA \
       -save_path=$BINARY_PATH \
       -context_window_width=$CONTEXT_WINDOW_SIZE \
       -scaling_coeff=$SCALING_COEFF \
       -sigma_coeff=$SIGMA_COEFF \
       -kappa_psi=$KAPPA_PSI \
       -top_n_word=$TOP_N_WORD \
       -start_year=0 \
       -end_year=$NUM_TIMES \
       -year_interval=1 \
       -num_iteration=$NUM_ITERATION > $LOG_PATH

# output probabilities
./prob -model_path=$BINARY_PATH -use_npmi=false > scripts/out/${SUFFIX}
cd scripts
python3 plot_prob.py ${SUFFIX}
cd ..
# parameters
# sampler
NUM_TIMES=10
NUM_SENSES=$1
VOCAB_SIZE=10000
NUM_SAMPLE=10000
SHIFT_TYPE="random"
WORD_PRIOR_TYPE="zipf"

# iscan
NUM_ITERATION=1000
KAPPA_PHI=10.0
KAPPA_PSI=100.0
TOP_N_WORD=$2

DAY=$(date "+%m%d")
INPUT_DATA=./data/pseudo_v3/pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_vocab${VOCAB_SIZE}_sample${NUM_SAMPLE}.txt
SUFFIX=pseudo_${WORD_PRIOR_TYPE}_sense${NUM_SENSES}_sample${NUM_SAMPLE}_vocab${VOCAB_SIZE}_kappa_phi${KAPPA_PHI}_kappa_psi${KAPPA_PSI}_top_n${TOP_N_WORD}_${DAY}
BINARY_PATH=./results/bin/${SUFFIX}.model
LOG_PATH=./results/log/${SUFFIX}

# generate pseudo data
python3 tools/create_pseudo_data.py --num-times $NUM_TIMES \
                                    --num-senses $NUM_SENSES \
                                    --vocab-size $VOCAB_SIZE \
                                    --num-sample $NUM_SAMPLE \
                                    --shift-type $SHIFT_TYPE \
                                    --word-prior-type $WORD_PRIOR_TYPE \
                                    --output-path $INPUT_DATA
# execute test
./scan -data_path=$INPUT_DATA \
       -save_path=$BINARY_PATH \
       -kappa_phi=$KAPPA_PHI \
       -kappa_psi=$KAPPA_PSI \
       -top_n_word=$TOP_N_WORD \
       -start_year=0 \
       -end_year=$NUM_TIMES \
       -year_interval=1 \
       -num_iteration=$NUM_ITERATION > $LOG_PATH

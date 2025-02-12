#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  for var in "$@"
  do
    case $var in
      --config=*)
          config=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)    
      ;;
      --eval_data=*)
          eval_data=$(echo $var |cut -f2 -d=)    
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {

    python main.py \
            --input_model ${input_model} \
            --config ${config} \
            --benchmark \
            --mode ${mode} \
            --eval_data ${eval_data}
}

main "$@"

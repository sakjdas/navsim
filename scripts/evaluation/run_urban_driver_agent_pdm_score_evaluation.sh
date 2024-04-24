SPLIT=mini
CHECKPOINT="/home/hguo/e2eAD/navsim_workspace/exp/training_urban_driver_pytorch/2024.04.24.18.55.53/pt_models/model1.ckpt"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=urban_driver_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=urban_driver_agent \
split=$SPLIT \

TEAM_NAME="SmallFish"
AUTHORS="Hao"
EMAIL="hao.guo@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=urban_driver_agent \
split=private_test_e2e \
scene_filter=private_test_e2e \
experiment_name=submission_urban_driver_agent \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
agent.checkpoint_path=/home/hguo/e2eAD/navsim_workspace/exp/training_urban_driver_pytorch/2024.04.24.23.49.31/pt_models/model1.ckpt \

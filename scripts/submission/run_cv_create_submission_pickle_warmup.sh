TEAM_NAME="SmallFish"
AUTHORS="Hao"
EMAIL="hao.guo@tum.de"
INSTITUTION="TUM"
COUNTRY="Germany"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
agent=constant_velocity_agent \
split=mini \
scene_filter=warmup_test_e2e \
experiment_name=submission_cv_agent_warmup \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \

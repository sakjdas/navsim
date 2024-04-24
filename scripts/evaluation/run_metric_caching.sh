SPLIT=mini

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
split=$SPLIT \
cache.cache_path=$NAVSIM_EXP_ROOT/metric_cache \
scene_filter.frame_interval=1 \

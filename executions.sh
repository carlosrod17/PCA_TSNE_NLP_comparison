log=/opt/shared/logs/log_$(date +"%Y%m%d_%H%M%S").log

python /opt/shared/code/Python/1_PROCESSING.py ${log}
python /opt/shared/code/Python/2_DIMENSIONALITY_REDUCTION.py ${log}
python /opt/shared/code/Python/3_1_CLUSTERING_MODELS_WITH_K_FIXED.py ${log}
python /opt/shared/code/Python/3_2_CLUSTERING_MODELS_WITH_K_OPTIMIZED.py ${log}
python /opt/shared/code/Python/3_3_CLUSTERING_MODELS_WITH_DBSCAN.py ${log}
python /opt/shared/code/Python/4_1_GET_INTERACTIVE_FIGURES.py ${log}
python /opt/shared/code/Python/4_2_GET_TFM_FIGURES.py ${log}
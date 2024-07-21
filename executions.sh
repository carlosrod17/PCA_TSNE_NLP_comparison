log=/opt/shared/logs/log_$(date +"%Y%m%d_%H%M%S").log

# python /opt/shared/code/1_PROCESSING.py ${log}
# python /opt/shared/code/2_DIMENSIONALITY_REDUCTION.py ${log}
# python /opt/shared/code/3_1_CLUSTERING_MODELS_WITH_K_FIXED.py ${log}
# python /opt/shared/code/3_2_CLUSTERING_MODELS_WITH_K_OPTIMIZED.py ${log}
# python /opt/shared/code/3_3_DBSCAN_MODELS.py ${log}
# python /opt/shared/code/4_1_GET_INTERACTIVE_FIGURES.py ${log}
python /opt/shared/code/4_2_GET_TFM_FIGURES.py ${log}
#nnssl_train 43 onemmiso -tr SimCLRTrainer_IntraSample_NC16_BS4_ep150 -p nnsslPlans -num_gpus 4
# nnssl_train 43 onemmiso -tr SimCLRTrainer_BS32_ep150 -p nnsslPlans -num_gpus 4
#nnssl_train 43 onemmiso -tr BaseMAETrainer_BS8_ep150 -p nnsslPlans -num_gpus 4
nnssl_train 43 onemmiso -tr VoCoTrainer_BS8_ep150 -p nnsslPlans -num_gpus 4
nnssl_train 43 onemmiso -tr BaseEvaMAETrainer_BS8 -p nnsslPlans -num_gpus 4
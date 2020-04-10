# export PYTHONPATH := ""

.PHONY: install
install:
# 	echo "You need to run all of the following commands:"
# 	echo "source activate py3"
# 	echo "export PYTHONPATH=\"${PYTHONPATH}\""
	pip install -Ue .

.PHONY: clean
clean:
	rm -rf ./logs
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete

.PHONY: help
help:
	python ./cirtorch/examples/test.py -h

.PHONY: run-benchmarks
run-benchmarks: clean
	-mkdir ./logs
	make resnet-gem > ./logs/resnet-gem.log
	make vgg-gem > ./logs/vgg-gem.log
	make pretuned-resnet-gem > ./logs/pretuned-resnet-gem.log
	make pretuned-vgg-gem > ./logs/pretuned-vgg-gem.log
	make resnet-rmac > ./logs/resnet-rmac.log
	make vgg-rmac > ./logs/vgg-rmac.log
	make resnet-mac > ./logs/resnet-mac.log
	make vgg-mac > ./logs/vgg-mac.log
	make resnet-spoc > ./logs/resnet-spoc.log
	make vgg-spoc > ./logs/vgg-spoc.log

.PHONY: resnet-gem
resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-gem' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: vgg-gem
vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-gem' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: pretuned-resnet-gem
pretuned-resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: prewhitened-resnet-gem
prewhitened-resnet-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-resnet101-gem' --multiscale --datasets 'scores' --whitening 'retrieval-SfM-120k'

.PHONY: pretuned-vgg-gem
pretuned-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: prewhitened-vgg-gem
prewhitened-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path 'retrievalSfM120k-vgg16-gem' --multiscale --datasets 'scores' --whitening 'retrieval-SfM-120k'

.PHONY: resnet-rmac
resnet-rmac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-rmac' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: vgg-rmac
vgg-rmac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-rmac' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: resnet-mac
resnet-mac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-mac' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: vgg-mac
vgg-mac:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-mac' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: resnet-spoc
resnet-spoc:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'resnet101-spoc' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: vgg-spoc
vgg-spoc:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-offtheshelf 'vgg16-spoc' --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: tuned-vgg-gem
tuned-vgg-gem:
	python ./cirtorch/examples/test.py --gpu-id '0' --network-path ./weights/scores_vgg16_gem_whiten_contrastive_m0.85_adam_lr1.0e-06_wd1.0e-04_nnum2_qsize250_psize2500_bsize1_imsize1024/model_epoch100.pth.tar --multiscale --datasets 'scores' --whitening 'scores'

.PHONY: train-vgg-gem
train-vgg-gem:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'scores' --test-datasets 'scores' --arch 'vgg16' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 2 --query-size=250 --pool-size=2500 --batch-size 1 --image-size 1024 --whitening --test-whiten 'scores'

.PHONY: train-example-vgg-gem
train-example-vgg-gem:
	python ./cirtorch/examples/train.py ./weights --gpu-id '0' --training-dataset 'retrieval-SfM-120k' --test-datasets 'roxford5k,rparis6k' --arch 'vgg16' --pool 'gem' --loss 'contrastive' --loss-margin 0.85 --optimizer 'adam' --lr 1e-6 --neg-num 5 --query-size=2000 --pool-size=20000 --batch-size 5 --image-size 362 --whitening

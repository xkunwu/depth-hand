# Hand in Depth

> Single view depth image based hand detection and pose estimation.

## Install

```
```

## Usage

```
cd code
python -m train.evaluate \
    --data_root=/data \
    --out_root=/output \
    --max_epoch=1 --batch_size=5 --bn_decay=0.9 \
    --show_draw=True --model_name=base_clean
```

## Tracking
If you happen to have a Intel® RealSense™ depth camera.
The code is tested using SR300 and D450.
Note: the code was updated to use [Intel® RealSense™ SDK 2.0](https://github.com/IntelRealSense/librealsense), so the minimum supported camera model is SR300.

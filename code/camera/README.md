# Hand detection and tracking

> Experimental hand detection and tracking from single depth camera.

<span style="display:block;text-align:center">![Test sequence.](/eval/test_seq.gif)</span>

Note: this is a two-week quick patch following the [hand pose estimation project](https://github.com/xkunwu/depth-hand) located in the main repo.
The purpose is to make live capture and hand tracking possible, but currently no plan to make the code waterproof (might be a future research project).

## Hardware prerequisite
The code was tested using SR300 and D400 series (D415/D435).
Please follow the [official instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) to install necessary drivers.

Note: the code was updated to use [Intel® RealSense™ SDK 2.0](https://github.com/IntelRealSense/librealsense), so the minimum supported camera model is SR300.

Also note: SR300 should work nicely after installing the drivers (hit 'realsense_viewer' to test if anything went wrong). But for D400 series, additional firmware update is necessary: go to the [official site](https://downloadcenter.intel.com/download/28377/Latest-Firmware-for-Intel-RealSense-D400-Product-Family?v=t) and following instructions there ('realsense_viewer' will also find any firmware issues and show you the latest update link).

## Usage
Default is using the 'super_edt2m' model.
Please check the main repo for [finding the full list of models](/README.md#print-model-list).

Note: tracking code does not require the prepared data, only need to download your favorite pretrained model. See [README.md in the pose estimation part](/README.md#pretrained-models) for more details.
```
cd depth-hand/code
python -m camera.capture --model_name=super_edt2m
```
-   If you want to see more debug information:
    ```
    python -m camera.capture --model_name=super_edt2m --show_debug=True
    ```
-   Save detection results:
    ```
    python -m camera.capture --model_name=super_edt2m --save_det=True
    ```
-   Save raw stream data (for replay and test):
    ```
    python -m camera.capture --model_name=super_edt2m --save_stream=True --save_det=True
    ```
    NOTE: previous data will be overwritten!
-   Read saved raw stream data (instead of live capture):
    ```
    python -m camera.capture --model_name=super_edt2m --read_stream=True --save_det=True
    ```

## Documents
Please check the [project webpage](https://xkunwu.github.io/projects/depth-hand/depth-hand) to see how it works or why not working as promised.

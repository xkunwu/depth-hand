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

### Basic usage
```
cd depth-hand/code
python -m camera.capture \
    --data_root=$HOME/data \
    --out_root=$HOME/data/univue/output \
    --model_name=super_edt2m
```
Key responses:
-   q: quit the current window.
-   b: save a single frame.

### Other options
-   If you want to see more debug information:
    ```
    python -m camera.capture \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --show_debug=True \
        --model_name=super_edt2m
    ```
-   Save detection results:
    ```
    python -m camera.capture \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --save_det=True \
        --model_name=super_edt2m
    ```
-   Save raw stream data (for replay and test):
    ```
    python -m camera.capture \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --save_stream=True --save_det=True \
        --model_name=super_edt2m
    ```
    NOTE: previous data will be overwritten!
-   Read saved raw stream data (instead of live capture):
    ```
    python -m camera.capture \
        --data_root=$HOME/data \
        --out_root=$HOME/data/univue/output \
        --read_stream=True --save_det=True \
        --model_name=super_edt2m
    ```
    This is very useful for debug: after making changes to the code, replay the same sequence and see if your idea works better.
    For example, you can play with the [test capture sequence](https://pan.baidu.com/s/1dm8gTcEOO0GjW6U9SEH1gw) used for generating the teaser figure on the top.

## Algorithms
Please check the [project webpage](https://xkunwu.github.io/projects/depth-hand/depth-hand) to see how it works or why not working as promised.

## FAQ
##### Q: There is a long halt before the first detection show up?
The Tensorflow pretrained model is loaded on the first frame when something interesting show up in the range.
On my laptop it roughly takes 2 seconds.
After that, the detection should be almost as fast as the capture rate of your depth camera.

##### Q: When turned on 'show_debug', only two blank windows show up?
A: That *sometimes* happens with the "TkAgg" [backend of matplotlib](https://matplotlib.org/faq/usage_faq.html#what-is-a-backend), which is unfortunately the default.
> [Tkinter is Python's de-facto standard GUI (Graphical User Interface) package."](https://wiki.python.org/moin/TkInter)

So make sure [Tcl/Tk](http://www.tcl.tk/) is installed correctly on your system.
Or switching to another backend might help (See [code/camera/capture.py](/code/camera/capture.py) around the top), e.g. "Qt5Agg" backen is more stable (IMO), but *extremely* slow.

##### Q: Detection looks very unstable?
A: Mostly due to noise of depth image, which may relates to (in addition to algorithm complexity): hardware sensibility, lighting condition, reflective objects within range (in one funny test case, our tester's metal watch makes detection very random :joy:), etc.

##### Note: other FAQ related to the algorithm may be found at the [project webpage](https://xkunwu.github.io/projects/depth-hand/depth-hand).

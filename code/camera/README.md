# Hand detection and tracking
> Experimental hand detection and tracking from single depth camera.

## usage
```
cd code
python -m camera.capture
```
-   If you want to see more debug information:
    ```
    python -m camera.capture --show_debug=True
    ```
-   Save detection results:
    ```
    python -m camera.capture --save_det=True
    ```
-   Save raw stream data (for replay and test):
    ```
    python -m camera.capture --save_stream=True --save_det=True
    ```
    NOTE: previous data will be overwritten!
-   Read saved raw stream data (instead of live capture):
    ```
    python -m camera.capture --read_stream=True --save_det=True
    ```

## Assumptions
-   single hand
-   hand is the closest to the camera
-   no other objects within 'crop_range', default: 100mm-480mm
-   pre-trained model was targeted at left hand

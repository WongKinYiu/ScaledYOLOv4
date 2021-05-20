# Welcome

If you are here, it means that you must be looking for a way to Deploy Scaled Yolov4 model using CPU. Original implementation uses `mishcuda` on the model loading itself so it generates error. `yolov4-csp` is actually `Scaled Yolov4`. 

Steps:
1. Clone this repo
2. Navigate to `ScaledYOLOv4/`
3. You'll find it empty because all work is done on the `yolov4-csp` branch
5. On your terminal, do `git checkout yolov4-csp`
6. Store your model weights somewhere and edit the `weights` variable inside `API_deploy_CPU.py` (Optional)
7. run `python API_deploy_CPU.py`. It'll deploy a very very basic model on `flask`
8. Input the path to weights on terminal
9. Use `postman` or `requests` module to send the request at `localhost:5000/predict`. Check the port number first.
10. Results returned are list of lists in the form of `[ [x_min, y_min, x_max, y_ax, class, conf_score], [......], .....[...], ]` 


I can bet You've missed `step No 4` ;)

## Note:
Code for this API is built around a Single class model. Please change and tweak the code given in `detect.py` according to your needs.

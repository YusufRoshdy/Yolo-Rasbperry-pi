# Yolo-Rasbperry-pi

link to the models:
https://drive.google.com/file/d/1j2_oH5rFfYDcFJmgQF7z723M7aT4C0sD/view?usp=sharing

extract the models in the 'models' directory

## To automate the script to run on startup
- open the terminal
- run this command to make the `.sh` execurable: `chmod +x run.sh`
- run `crontab -e` to open the cron file
- add this to the file `@reboot <path to run.sh>`: put the abslute path of run.sh.
it should look somthing like `@reboot /home/pi/Desktop/Yolo-Rasbperry-pi/run.sh`
- save and exit


For the multiprocess code do the same steps as above, but instead of using `run.sh` use `mp_run.sh`

Note: the multiprocess code runs without the need to have the web streem open.
it even runs faster that the normal code when the web streem is closed.
On the other hand, when the web streem is open it runs slower because the overhead
of the inter process communication 


## custom models
To use a custom model, creat a directory called `custom` and add you models to it
the name of the model has to contain it's type (i.e. if the model is a yolov5
version then 'tolov5' has to be in the name).

Then restart the code to see the effect

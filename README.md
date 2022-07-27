# Yolo-Rasbperry-pi

link to the models:
https://drive.google.com/file/d/1j2_oH5rFfYDcFJmgQF7z723M7aT4C0sD/view?usp=sharing

extract the models in the 'models' directory

To automate the script to run on startup
- open the terminal
- run `crontab -e` to open the cron file
- add this to the file `@reboot <path to run.py>`: put the abslute path of run.sh.
it should look somthing like `@reboot /home/pi/Desktop/Yolo-Rasbperry-pi/run.sh`
- save and exit

# 1. Build Robot

Building blocks provided: Servo motors, bricks, hinges, heads
1. Align carefully to the robot in simulator, especially be careful on the direction of hinges. Disassemble parts is annoying and waste of time. 
	1. Make sure the rotational degree, and the placement of stators (longer) and motors(shorter) are aligned perfectly with the simulated robot. That is almost identical as saying that  the motors should follow "same conventions" (the part that connect wires always pointing out relative to the core, or towards the core). 
2. A microSD card with PiOS. 
3. Camera: 

Principle: 
1. Separate the servos to different assembly (there are 2), to avoid possible power/current-load problem (the hardware cannot deliver the requested simultaneous motion cleanly under load)
	1. E.g. Puts forelimb/left-front group on assembly 1 and spine/hind group on assembly 2
2. Connect the servos and modules AS TIGHT AS POSSIBLE, to make every part firmly connected. Otherwise during performing aggressive movement, it might break. 
# 2. RPi Setting Up
RobotHat, Raspberry pi
> Keep in mind:
> 1. Charging batteries is only permissible when you keep an eye on them. 
> 2. Don't left battery in the robot. The battery keeps discharging to the point that it starts messing with the actual robot until the head breaks. So please make sure to remove the batteries.

1. Flash the microSD card with Raspberry Pi OS 
	1. Download Raspberry Pi Imager, a quick tool to install Raspberry Pi OS and other operating systems to a microSD card.
		1. Device: `Raspberry 4`
		2. OS: `Raspberry pi OS (64 bits)
		3. Hostname: `pi`
		4. Username: `pi`;  Password: `pi`
		5. Keyboard layout: *us* ( It can always be changed using the raspi-config tool on the RPi)
		6. Wifi: connect to **Thymionet** (password:`172luckytulip75B`)
		7. Enable SSH, use password authentication
		8. Enable Raspberry Pi Connect.
2. SSH:
	1. Connect the laptop and the RPi to **Thymionet** 
	2. ==Find the ip address: `10.15.2.203`==
	3. Ssh to it in terminal: `ssh pi@IPADDRESS`
3. Clone the project's git repo (via SSH)
	1. Fix SSH certification issue in git clone
		 1. Date and time:
			 1. Check if the time and data is wrong: `date` 
			 2. Trigger a time sync: `sudo timedatectl set-ntp true`
		 2. Or:`sudo apt install -y htpdate fake-hwclock`, `sudo htpdate -s www.google.com`
		 3. Update certificate: `sudo apt update` and `sudo apt install --only-upgrade ca-certificates`
		 4. Authenticate pi with GitHub:
			 1. Generate an SSH Key on your Raspberry Pi: `ssh-keygen -t ed25519 -C "y.zhou12@student.vu.nl"`i
			 2. Copy the SSH key: `cat ~/.ssh/id_ed25519.pub`
		 5. Add the key to your GitHub account
			1. Go to GitHub and log in.
			2. In the top-right corner, click your profile picture and go to Settings.
			3. In the left sidebar, click SSH and GPG keys.
			4. Click the green New SSH key button.
			5. Give it a Title (e.g., "Raspberry Pi").
			6. Paste your copied key into the Key field.
			7. Click Add SSH key.
		6. Clone the private repo using SSH `g
	2. Don't download Ariel in the Pi! Also, don't `uv sync`!
4. Allow git commit and push
	1. Set: `git config --global user.email "your_email@example.com"`
	2. And: `git config --global user.name "Yizhou"`
5. Install uv and packages (for Harware)
	1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
	2. Install numpy&opency: `uv pip install numpy,opencv-python`
	3. Install cpu-only torch: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
	4. Verify [[picamera2]] installation ([documentation](https://pip-assets.raspberrypi.com/categories/652-raspberry-pi-camera-module-2/documents/RP-008156-DS-2-picamera2-manual.pdf)) in the RPi:
		   1. Check by `sudo apt install -y python3-picamera2`
		   2. To update *picamera2*, use `sudo apt update` followed by `sudo apt full-upgrade`
6. Install CI Group's *robot-hat* library ([git repo](https://github.com/sunfounder/robot-hat)) in the RPi:
	1. Download the repo: `sudo apt install -y git`, `cd ~`, `git clone git@github.com:ci-group/robohat.git`
	2. Update packages and fix the GPIO dependency: 
		1. ```bash
		   sudo apt update
		   sudo apt upgrade -y
		   sudo apt remove -y python3-rpi.gpio
		   sudo apt install -y python3-rpi-lgpio i2c-tools
		   ```
	   3. Then: `sudo cp /boot/firmware/config.txt /boot/firmware/config.txt.backup`, `sudo cp ~/robohat/setup_files/config.txt /boot/firmware/config.txt`, then `sudo reboot`
	   4. Put the files where the library expects them: 
		   ```bash
		   mkdir -p ~/bin ~/robohat
			cp ~/robohat/bin/robo ~/robohat/bin/servo ~/robohat/bin/buzz_random ~/bin/
			cp -r ~/robohat/robohatlib ~/robohat/
			cp -r ~/robohat/testlib   ~/robohat/
			cp ~/robohat/Test.py ~/robohat_repo/SerTest.py ~/robohat/
		   ```
	
- **Useful commands:**
	- Check remaining storage: `du -h --max-depth=1 ~ 2>/dev/null | sort -h | tail -20`, 
		- Check space occupied by uv: `du -sh ~/.cache/uv 2>/dev/null`
	- Clean the uv virtual environment: `uv cache clean` -> uv keeps a global wheel cache; this can be several GB; `rm -rf ~/ariel/.venv`  -> delete the bloated venv (adjust path to your ARIEL checkout)
	- Delete a directory:`rm -r PATH`
	- Copy file ==from Local to SSH==: ` scp /Users/yizhou/bachlor_project robohat@10.15.2.203:/home/robohat`
		- A folder: ` scp -r /Users/yizhou/bachlor_project robohat@10.15.2.203:/home/robohat`
	- Copy file from SSH to local: `scp PATH yizhou@10.15.2.8:/Users/yizhou/bachlor_project/A_TEMP/`
# 3. Test & Calibration
[[Baby Robot Configuration]]
Test:
1. Use `robo` and `servo` to test battery, camera, and servos. 

Calibration:
1. run `servo`, run 5 to move all servos to 90 degrees
2. Power off the servo rail, but do not rotate the servo shafts by hand.
3. Unscrew each servo horn/limb linkage that is not neutral.
4. Reattach the horn so the physical limb is as close as possible to neutral while the servo shaft is still at commanded 90.
5. For small remaining offsets, do manual calibration:
	1. Run command `robo`, then command `set servo angle [servo nr] [angle]`  repeatedly for every servo to find an angle that can make that servo go to visually neutral position.
		1. E.g. `set servo angle 0 85`, `set servo angle 0 95`.
	2. Edit baby_hardware.py in DEFAULT_SERVO_MAPPINGS -> neutral position.
Need re-calibration if the servos are plugged somewhere else. 
# 4. Common Issue
1. When can boost the OS by cable but not battery: check the connection of the yellow cable to both the Power Manager board and RPi. 
2. If the servos can be detected but cannot be manipulated to: check the 
3. Be careful: Don't use 270 degrees servo! Use exactly the same type (degrees), if possible, use the servos with exactly the same color, otherwise you'd need an extra calibration cable (no idea how to do). 
	1. These two types of servos require different direction connection 
4. `ssh: connect to host 10.15.2.203 port 22: Host is down`, or Operation timed out.
	1. Try connect to the monitor. If the OS cannot be successfully launched (a rainbow color square size appearing and disappearing repeatedly), try to charge the battery then try again. 
		1. I guess it is because the power supply of the battery is not stable/strong enough to supply the RPi. So replace the battery can help.
5. If ssh connection is not successful, first check if the laptop is connected to Thymionet before touching the robot.

How to turn power on: hold the small yellow button until the red light start constantly flickering, then immediately release it. Redo if not succeed. 


# Suggestions for hardware design
1. Can add an extra connection place in the hinge's rotor, two connections are not strong enough for aggressive movement in long arms/legs
2. Adding another two holes for screws at the robot's bottom lid. If only use 2, the robot is not stable enough. Also, adding screws will lift the robot's head a little bit, then the robot's limb cannot touch the ground anymore (like in the simulator). 
	1. Or in simulator, consider adding the screws' height.

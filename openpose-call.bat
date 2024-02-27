echo off


setlocal enabledelayedexpansion enableextensions
set LIST=
for %%x in (*.mp4) do set LIST=!LIST! %%x
set LIST=%LIST:~1%

cd C:/Users/manma/Desktop/openpose_gpu
for %%a IN (%LIST%) do (
	set b=%%a
	set b=!b:~0,7!
	call C:/Users/manma/Desktop/openpose_gpu/bin/OpenPoseDemo.exe --video C:/Users/manma/Desktop/eatingvideos/%%a --hand --tracking 1 --number_people_max 1 --write_json C:/Users/manma/Desktop/eatingvideos/!b!
)

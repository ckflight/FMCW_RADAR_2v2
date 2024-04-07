# FMCW_RADAR_2v2
Python process and analysis files for the updated second version of Frequency Modulated Continuous Wave Radar Hardware Design

I have designed the second version of FMCW Radar. In this design, i tried to improve dynamic range without using expensive components with a cost effective approach. 
I will not make pcb design files available for now because the design has a lot of details which required 1 year of search and development where i have tried lots of possibilities
such as affects of regulators, opamp circuit optimazation and gain stages, optimazation of RF circuit and paths.
Without using FPGA and high speed ADC (high sampling rate and decimation decreases noise floor)
I managed to have noise floor around -90 dBFS with antennas and -115 dBFS with RX TX ports are 50 ohm terminated.

I am posting the processing files i have done on host side with Python for anyone to learn.

## Capabilities of the radar:
 * Detection range up to 2km for building or terrain size objects and a few hundred meters for car like objects (i haven't checked the max distance for cars).
 * Detection of human in 120 meters in a hallway road with more than 50 cars, big trees throughout the road and buildings. This is such a noisy environment to detect small objects. (i haven't checked max distance for human size object at open area)
 * Detection of movements as small as 75 micrometers with phase measurement. Heartbeat, breathing like movements up to 2.5cm is detectable for a stationary object.
 * Detection of movements and stationary objects at closed area and short distances as small as a few meters.
 * Frequency synthesis with both PLL or DAC with analog switch for Frequency Modulated or Doppler measurement.
 * SDIO microsd card interface for logging without host.

![Untitled5](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/5fa3c864-8e84-449e-b6a0-8482ac3ec935)
   
## Radar_Plot_QT.py
It plots the radar data with Python Qt5 and Vispy plotting tool which uses GPU instead of CPU. Code plots the time domain, frequency domain, dBFS noise floor and phase through the whole sampling frequecy spectrum together with scale and restart options. It has thread based approach between Vispy and Qt classes by using DataSource which allows us to manage more data. The code has different filtering options and also can plot faster by decreasing frequency range.

![Untitled](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/cec61433-47c9-48c4-b16d-7eb0aa7097c5)

## Radar_Plot_Phase.py 
It makes phase analysis by focusing on an object at specific location to measure small movements like breathing and heartbeat.
 * The part between 4sec and 14 is slow and fast breathing measurement.
 * Rest of the plot belongs to my heartbeats.

![Untitled2](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/2a77585c-c142-442f-9b75-dcc71aa03fbb)

## Radar_Plot_WaterFall.py
It plots the colormesh of the radar data for better visualisation. It has option of removing the clutter as well.
 * First plot belongs to me walking in the hallway. I removed stationary objects.
 * Second plot is the long range measurement from my home's terrace. It detects buildings at 400m, 700m and 1.3km
![Untitled2](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/5ac43d7a-e37b-4f26-bcc8-ff13260f47a8)
![Untitled](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/b47b74a9-1097-497e-ab63-a60b8339eab3)

## Radar_Plot_SAR.py
This file generates the Synthetic Aparture image. I am still working on this part and it will be updated as i make progress.
I need to add omega_k algorithm for focusing the image for better resolution.

![Untitled3](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/69b53866-90ac-4cb7-8b86-18fcbdcdc29c)

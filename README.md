# FMCW_RADAR_2v2
Python process and analysis files for the updated v2.2 of Frequency Modulated Continuous Wave Radar Hardware Design

I have designed the new version of FMCW Radar. In this design, i tried to improve receiver chain and adc dynamic range without using expensive components with a cost-effective approach. I will not make pcb design files available for this design because it has a lot of details which required 1 year of search and development where i have tried lots of possibilities such as effects of regulators, Op-Amp circuit optimization and gain stages, optimization of RF circuit and paths.
Without using FPGA and high speed ADC (high sampling rate and decimation decreases noise floor)
I managed to have noise floor around -90 dBFS with antennas and -115 dBFS with RX and TX ports are 50 ohm terminated.

I am posting the processing files i have done on host side with Python for anyone to learn.

## Capabilities of the radar:
 * Detection range up to 2 km for building or terrain size objects.
 * Detection of cars in 1 km from home view along with buildings and trees etc. (which is a very noisy environment for detection).
 * Detection of human in few hundred meters in a hallway road with more than 50 cars, big trees throughout the road and buildings. This is such a noisy environment to detect small objects. (I haven't checked max distance for human size object at an open area)
 * Detection of movements as small as 75 micrometers with phase measurement. Heartbeat, breathing like movements up to 2.5 cm is detectable for a stationary object.
 * Detection of movements and stationary objects at closed area and short distances as small as a few meters.
 * Frequency synthesis with both PLL or DAC with analog switch for Frequency Modulated or Doppler measurement.
 * SDIO Micro SD Card interface for logging without host. (Faster logging is important in the case of decreasing the ramp delay between sweeps which improves the PLL step number which affects the frequency hopping resolution by making smaller steps between transitions)
 * In this design, i have added an Op-Amp auto gain stage, but fix gain makes it less noisy so it is available as well.
 * PCB stack-up has changed and i have re-designed the directional coupler which works better compared to the previous version with better directivity.
 * Port are better matched compared to the previous version which decreases reflection. Therefore, power is efficiently transferred without any loss. In this design radar can benefit from maximum gain without any loss.

![Untitled5](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/5fa3c864-8e84-449e-b6a0-8482ac3ec935)
   
## Radar_Plot_QT.py
It plots the radar data with Python Qt5 and Vispy plotting tool which uses GPU instead of CPU. Code plots the time domain, frequency domain, dBFS noise floor and phase through the whole sampling frequecy spectrum together with scale and restart options. It has thread based approach between Vispy and Qt classes by using DataSource which allows us to manage more data. The code has different filtering options and also can plot faster by decreasing frequency range.

![Untitled](https://github.com/ckflight/FMCW_RADAR_2/assets/61315249/cec61433-47c9-48c4-b16d-7eb0aa7097c5)

## Radar_Plot_Phase.py 
It makes phase analysis by focusing on an object at specific location to measure small movements like breathing and heartbeat.
 * 6 to 12 sec: Slow breathing
 * 11 to 16 sec: Fast breathing
 * 17 to 20 sec: I holded my breath so pulses are detected
 * 20 to 22 sec: A deep breath
 * Rest of the plot belongs to my heartbeats

![Untitled2](https://github.com/ckflight/FMCW_RADAR_2v2/assets/61315249/c3ede918-5830-4b8d-8c96-310ad28ab3b0)

## Radar_Plot_WaterFall.py
It plots the colormesh of the radar data for better visualisation. It has option of removing the clutter as well.
 * First plot: Home view with cars detected in km range.
 * Second plot: Me walking in a parking lot among lots of cars. There are trees from 100 to 130 m. There is a little waterway from 130 to 140 m so there might be some ghosting articacts around it.
![Untitled](https://github.com/ckflight/FMCW_RADAR_2v2/assets/61315249/b51daa43-6c51-4719-9d20-c8e37f4b8ec3)
![Untitled1](https://github.com/ckflight/FMCW_RADAR_2v2/assets/61315249/33c41898-376a-49c4-84ee-9779a1b87801)

## Radar_Plot_SAR.py
This file generates the Synthetic Aparture image. I am still working on this part and it will be updated as i make progress.
I need to add omega_k algorithm for focusing the image for better resolution.

![Untitled3](https://github.com/ckflight/FMCW_RADAR_2v2/assets/61315249/0eee0846-5074-4150-987e-e8686c60e892)

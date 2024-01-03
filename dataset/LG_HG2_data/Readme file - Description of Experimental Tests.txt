The included tests were performed at McMaster University in Hamilton, Ontario, Canada by Dr. Phillip Kollmeyer (phillip.kollmeyer@gmail.com).  If this data is utilized for any purpose, it should be appropriately referenced.

A brand new 3Ah LG HG2 cell was tested in an 8 cu.ft. thermal chamber with a 75amp, 5 volt Digatron Firing Circuits Universal Battery Tester channel with a voltage and current accuracy of 0.1% of full scale.

/*************************************************/
A series of tests were performed at six different temperatures, and the battery was charged after each test at 1C rate to 4.2V, 50mA cut off, with battery temperature 22degC or greater.  The tests were performed as follows:
1. Four pulse discharge HPPC test (1, 2, 4, and 6C discharge and 0.5, 1, 1.5, and 2C charge, with reduced values at lower temperatures) performed at 100, 95, 90, 80, 70..., 20, 15, 10, 5, 2.5, 0 % SOC.   
2. C/20 Discharge and Charge test.
3.0.5C, 2C, and two 1C discharge tests.  The first 1C discharge test is performed before the UDDS cycle, and the second is performed before the Mix3 cycle.
4.Series of four drive cycles performed, in following order: UDDS, HWFET, LA92, US06.  
5. A series of eight drive cycles (mix 1-8) consist of random mix of UDDS, HWFET, LA92, US06. The drive cycle power profile is calculated for a single LG HG2 cell in a compact electric vehicle.
6. The previous tests are repeated for ambient temperatures of 40degC, 25degC, 10degC, 0degC, -10degC, and -20degC, in that order. For tests with ambient temperature below 10degC, a reduced regen current limit is set to prevent premature aging of the cells. The drive cycle power profiles are repeated until 95% of the 1C discharge capacity at the respective temperature has been discharged from the cell.

/*************************************************/
Matlab (.mat) file naming convention:
The file names begin with the date the test started, and are followed by the time the test started and a descriptive name, such as in the following example:
"11-07-18_10.49 557_Cap_1C_40degC_LGHG2.mat”

Date: "11-07-18" (November 7th, 2018)
Time: "10.49" (10:49 am)
Descriptor: " Cap_1C_40degC_LGHG2"

The naming convention allows for sorting by date and test time when all the files are placed in one folder.  This may be useful when trying to determine which charge is associated with which drive cycle, for example, or when looking at aging.

/*************************************************/
Data columns:
Time (time in seconds)
TimeStamp (timestamp in MM/DD/YYYY HH:MM:SS AM format)
Voltage (measured cell terminal voltage, sense leads welded directly to battery terminal)
Current (measure current in amps)
Ah (measured amp-hours, with Ah counter typically reset after each charge, test, or drive cycle)
Wh (measured watt-hours, with Wh counter reset after each charge, test, or drive cycle)
Power (measure power in watts)
Battery_Temp_degC (battery case temperature, at middle of battery, in degrees Celsius measured with a AD592 +/-1degC accuracy temperature sensor)
/*************************************************/
Time step:
Tests considered more important, such as drive cycles, were saved with a 0.1 second time step.  Other test portions, such as charges and pauses, were considered to have slower dynamics and be less important, and were therefore save at a lower data rate.  Be aware of these variances in data rate, and up sample the logged data if necessary to achieve a higher or consistent data rate.  Files may also have variable sample rates, so be sure to plot time versus the parameter of interest.

CSV files:
The original CSV files saved by the cycler are also included, they have data identical to that in the .mat files.  The files are labeled with test number and a test description.  For example "555_Dis_2C" is test 555 as assigned by the cycler and a 2C discharge.  The header in the files contains additional descriptive information.

/*************************************************/


## Folder Structure

```
SeaIce-Extent-Prediction-ML/
├─ data/
|  ├─ c02_annmean_gl.csv   #Global marine surface CO2 concentration (annual)
|  ├─ c02_mm_gl.csv        #Global marine surface CO2 concentration (monthly)
|  ├─ N_seaice_extent_dailyv4.0.csv   #North sea-ice extent 
|  └─ NH.Ts+dSST.csv                  #North land-ocean surface temperature anamolies
├─ notebooks/            
|  ├─ data_analysis.ipynb
|  ├─ first_ice_free_year_prediction.ipynb          #Predicting the first "ice-free" year using a linear model
|  └─ SIE_prediction.ipynb                          #Predicting SIE using different models
├─ results/                  #Results of the predictions   
│  ├─ first_ice_free_year.csv
│  └─ SIE_prediction.csv
├─ source/                    #Code for the respective processes
│  ├─ models.py
│  ├─ preprocessing.py
|  ├─ saving_results.py
│  └─ visualisation.py             
└─ README.md
```




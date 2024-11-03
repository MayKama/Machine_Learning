Damaturu Project is on impact of climate change on malaria outbreak
In this study, unsupervised machine learning techniques were leveraged and evaluated using root mean squared error (RMSE) in understanding the nexus between climatic variables (rainfall and temperature) and its impacts on the prevalence of malaria in Damaturu, Yobe State, Nigeria. Considering the continuous nature of the target data (malaria), predictive techniques like Linear Regression, Lasso Cross Validation (Lasso CV), Ridge and TheilSen Regression were fitted in predicting malaria cases, given rainfall and temperature (and other engineered features) with their respective evaluation metrics compared acrossboard. This was to help with ensuring consistency and the comprehensive efficiency of results from the predictive model. We further discuss the data collection, preprocessing, and feature extraction using the dataset obtained.
Behind fitting the various predictive models on the train set (training data) and evaluating their corresponding performance on the test set, the Lasso CV model was observed to have exhibited the least error (RMSE value) when comparing actual malaria cases from the predicted, and eventually identified as the most effective method for the prediction of the malaria outbreak in our area of study.  The analysis and prediction of malaria using the identified climate variables is significant for early detection and control of malaria outbreak, and towards enhancing resilience and health of people in the target location. 

3.1 Data collection
This study used decade-long (2014-2023) datasets collected from reliable and relevant sources in Yobe State. The malaria data were collected from the epidemiologic data repository of the state, with the approval of Yobe State Primary Healthcare Management Board, while the climate data (rainfall and temperature) were sourced from the meteorological station of Yobe State University, Damaturu, Nigeria. These institutions are widely known for managing reliable and accurate datasets, thus sourced and used for this study.

3.2 Data Overview
Three independent secondary datasets of Damaturu town were collected from two sources in the study area, consisting of rainfall, temperature and malaria. Rainfall data comprises 120 observations (rows) and three features (columns) namely: Year, Month and Rainfall. Temperature data had a similar number of observations and features namely: Year, Month and Temperature. While the malaria data similarly exhibited the same number of rows and columns namely: Year, Month and Malaria. Rainfall, Temperature and Malaria were reported in monthly averages across the year and recorded in millimeters, degree celsius and thousands, respectively. These independent data were subsequently merged along the Year and Month column (since these 2 features were common and consistent across all three columns) to obtain a new dataset comprising 120 rows and 5 features namely: Year, Month, Rainfall, Temperature and Malaria – to determine which of the climatic factor is a major contributor to human health as well to determine what factors based on the features militate against the human health thereby helping health practitioners make decision on what to do and what advice to give to their patients ahead of time.

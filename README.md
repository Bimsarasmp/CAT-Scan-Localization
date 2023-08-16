# CT-Scan-Localization

Cat scan localization in medical imaging is an essential aspect of contemporary healthcare. Computed Tomography (CT) scans provide detailed cross-sectional images of the human body, and it is of the utmost importance to precisely determine the relative location of each CT slice on the axial axis. This endeavor facilitates accurate anatomical mapping, which aids in the diagnosis, treatment planning, and monitoring of a variety of medical conditions.

The dataset used in this analysis is the "Relative location of CT slices on axial axis" dataset, obtained from the UCI Machine Learning Repository. This dataset consists of CT scan images represented as 2D matrices, with each matrix corresponding to a CT slice.

Dataset : 
https://archive.ics.uci.edu/dataset/206/relative+location+of+ct+slices+on+axial+axis

we will take “reference’ as target variable and we will estimate the relative location of CT slices on the axial axis. 

Field Descriptions:
•	patientId: Each ID identifies a different patient
•	value[1-241]: Histogram describing bone structures
•	value[242 - 385]: Histogram describing air inclusions
•	386: Relative location of the image on the axial axis (class value).
Values are in the range [0; 180] where 0 denotes the top of the head and 180 the soles of the feet.



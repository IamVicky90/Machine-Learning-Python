from joblib import dump, load
import numpy as np
model=load("Real_estate_py.joblib")
test=np.array([-0.43942006,  360.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23775409, -14.31238772,  5.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034])
print(model.predict([test]))
# -*- coding: utf-8 -*-
import pandas as pd

stars = pd.read_csv("/Users/anuraagchandra/Downloads/stars_1000.csv")

X = stars[[
    "temperature_K",
    "luminosity_solar",
    "radius_solar",
    "color_index"

]]

y = stars["spectral_type"]

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

temp_str = input("Temperature:")
temp = float(temp_str)
lum_str = input("Luminosity:")
lum = float(lum_str)
rad_str = input("Radius:")
rad = float(rad_str)
col_str = input("Color Index:")
col = float(col_str)

new_star = {
"temperature_K" : temp,
"luminosity_solar" : lum,
"radius_solar" : rad,
"color_index" : col
}

import pandas as pd

new_star_df = pd.DataFrame([new_star])
prediction = model.predict(new_star_df)
print(prediction)

stars2 = pd.read_csv("/Users/anuraagchandra/Downloads/stars_10000_unlabeled.csv")
X_NEW=stars2[[
    "temperature_K",
    "luminosity_solar",
    "radius_solar",
    "color_index"

]]
predictions = model.predict(X_NEW)
print(predictions)

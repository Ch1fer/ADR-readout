from NN_analog_clock_reader import predict_clock as pc
from photo_preprocessing import preprocessing 

file_path = "photo_preprocessing/assets/clock4.jpg"
preprocessing.draw(file_path)

output_image = preprocessing.draw(file_path)

pc.predict(file_path)
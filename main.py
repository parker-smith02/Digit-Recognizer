from model import *
from image_processing import crop_images, format_numbers, image_to_np_array
import os



#model creation and training
data = prepare_data()

model = create_model(True)

model = compile_model(model)

model = train_model(model, data)

evaluate_model(model, data)


#preparation of images
crop_images()

format_numbers()


#predict numbers
guess = ''
for file in os.listdir('numbers'):
    if file.endswith('.jpg'):
        array = image_to_np_array(file)
        array = format_array(array)
        prediction = model.predict(array)
        prediction_1 = np.amax(prediction)
        digit = np.where(prediction_1 == prediction)
        digit = digit[1][0]
        guess = guess + str(digit)
    
print(guess)


















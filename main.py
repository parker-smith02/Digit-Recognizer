from model import *

data = prepare_data()

model = create_model(True)

model = compile_model(model)

model = train_model(model, data)

evaluate_model(model, data)

visualize_output(model, data)
<<<<<<< HEAD

=======
>>>>>>> f4cb3b0b9b46306aa6328cbc0462bc9b1ba40d2a











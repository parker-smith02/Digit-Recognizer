from model import *

data = prepare_data()

model = create_model(True)

model = compile_model(model)

model = train_model(model, data)

evaluate_model(model, data)

visualize_output(model, data)











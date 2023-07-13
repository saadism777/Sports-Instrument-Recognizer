from fastai.vision.all import *
import gradio as gr

sport_instrument_labels = (
'Badminton cork',
'Badminton racket',
'Baseball bat',
'Basket ball',
'Bowling ball',
'Cricket ball',
'Cricket bat',
'Cricket helmet',
'Frisbee disc',
'Goal keeping gloves',
'Golf club', 'Hockey helmet',
'Hockey stick', 'Lacrosse stick',
'Pool cue',
'Rugby ball',
'Soccer ball',
'Squash racket',
'Table tennis paddle',
'Tennis racket',
'Volleyball ball'
)

model = load_learner('models/dataloader-v1.pkl')

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(sport_instrument_labels, map(float, probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'test_images/ex1.jpg',
    'test_images/ex2.jpeg',
    'test_images/ex3.jpg',
    'test_images/ex4.jpg',
    'test_images/ex5.jpg',
    'test_images/ex6.jpg',
    'test_images/ex7.jpg',
    'test_images/ex8.jpg',
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)
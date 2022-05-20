from tensorflow.keras.models import load_model
    
model = load_model("cats_and_dogs_small_pretrained.h5")
model.summary()
conv_base = model.layers[0]
conv_base.summary()
conv_base.trainable = False
print("="*30)
print("="*30)
print("="*30)
print("="*30)
print("="*30)
print("="*30)
print("="*30)
for layer in conv_base.layers[:249]:
    layer.trainable = False
for layer in conv_base.layers[249:]:
    layer.trainable = True

conv_base.summary()
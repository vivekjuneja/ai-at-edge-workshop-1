# experiments with object recognition
Ensure PYTHONPATH is set to the models and object_detection directory. TENSORFLOW below refers to the containing directory of the Tensorflow models  
```
export PYTHONPATH="$PYTHONPATH:$TENSORFLOW_DIR/models/research/:$TENSORFLOW_DIR/models/research/object_detection"
```

Ensure Tensorflow python packages are installed before running the demo. 

```
python3 recognition.py ./test_images/image3.jpg
```

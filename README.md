# Handwritten digit Recognizer

A GUI app made with ralib where you can draw a digit and model tries to recognise the digit you drew.

install raylib from https://www.raylib.com/

compile and launch GUI with

``` bash
g++ -I <your_raylib_include_path> -L <your_raylib_lib_path> -o main index.cpp -lraylib -lgdi32 -lopengl32 -lwinmm
./main
```


### Custom Training
- open train.cpp
- tweak the hyperparameters
- (optionally) update the weights output file path.

compile and start training with

``` bash
g++ train.cpp -o train
.\train
```

### Testing
update the weights output file path.

compile and run with

``` bash
g++ test.cpp -o test
.\test
```

import 'dart:io';

import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Prediction App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const PredictionPage(title: 'Dog Cat Prediction App'),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key, required this.title});

  final String title;

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  late Interpreter interpreter;
  File? _imageFile;
  List? _listResults;
  String? _classificationResult;

  @override
  Widget build(BuildContext context) {

    loadModel();

    return Scaffold(
        appBar: AppBar(
          title: Text(widget.title),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              Container(
                margin: const EdgeInsets.all(15),
                padding: const EdgeInsets.all(5),
                decoration: const BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.all(Radius.circular(0.1)),
                  shape: BoxShape.rectangle,
                  boxShadow: [
                    BoxShadow(
                      offset: Offset(1, 1),
                      color: Colors.black12
                    )
                  ],
                ),
                child: (_imageFile != null) ? Image.file(_imageFile!) : Image.network('https://i.imgur.com/sUFH1Aq.png'),
                ),
              Text(
                  (_classificationResult != null) ? _classificationResult! : "",
                  style: const TextStyle(
                    fontSize: 50,
                  ),
              ),
              ElevatedButton(onPressed: loadImage,
                  child: const Icon(Icons.camera))
            ],
          )
        ),
    );
  }

  Future loadModel() async {
    interpreter = await Interpreter.fromAsset('model_catdog.tflite');
    debugPrint(interpreter.toString());
  }

  Future loadImage() async {
    var imagePicker = ImagePicker();
    var image = await imagePicker.pickImage(source: ImageSource.gallery, maxHeight: 300);
    var imgFile = File(image!.path);
    classifyImage(img.decodeImage(imgFile!.readAsBytesSync())!);
    setState(() {
      _imageFile = File(image!.path);
    });
  }

  Future classifyImage(img.Image image) async {
    var inputShape = interpreter.getInputTensor(0).shape;
    var inputType = interpreter.getInputTensor(0).type;

    ImageProcessor processor = ImageProcessorBuilder()
        .add(ResizeOp(inputShape[1], inputShape[2], ResizeMethod.NEAREST_NEIGHBOUR)).build();

    TensorImage tensorImage = TensorImage(inputType);
    tensorImage.loadImage(image);
    tensorImage = processor.process(tensorImage);
    var output = List.filled(2, 0).reshape([1, 2]);
    interpreter.run(tensorImage.buffer, output);
    debugPrint(output.toString());
    if (output[0][0] > output[0][1]) {
      _classificationResult = "Cat";
    } else {
      _classificationResult = "Dog";
    }
  }

}

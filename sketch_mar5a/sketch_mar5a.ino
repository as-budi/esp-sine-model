#include "EloquentTinyML.h"
#include "eloquent_tinyml/tensorflow.h"

#include "sine_model.h"

#define IN 1
#define OUT 1
#define ARENA 4096

Eloquent::TinyML::TensorFlow::TensorFlow<IN, OUT, ARENA> tf;

void setup() {
    Serial.begin(9600);
    tf.begin(sine_model);
}

void loop() {
  for (int i = 0; i < 200; i++){
    float x = 3.14 * i / 100.0f;
    float input[1] = { x };
    float y_true = 3 * sin(x);
    float y_pred = 3 * tf.predict(input);

    Serial.print("Variable_1:");
    Serial.print(y_true);
    Serial.print(",");
    Serial.print("Variable_2:");
    Serial.println(y_pred);
    // Serial.print("sin(");
    // Serial.print(x);
    // Serial.print(") = ");
    // Serial.print(y_true);
    // Serial.print(",");
    // Serial.print("predicted: ");
    // Serial.println(y_pred);
    // delay(1000);
  }
    
}

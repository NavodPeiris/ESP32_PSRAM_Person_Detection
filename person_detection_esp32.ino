// #include "Arduino.h"
#include <TensorFlowLite_ESP32.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <esp_task_wdt.h>

#include "person_detect_model_data.h"
#include "model_settings.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <esp_heap_caps.h>

//#include "img_converters.h"
//#include "Free_Fonts.h"

#include "soc/soc.h" // Disable brownout problems
#include "soc/rtc_cntl_reg.h" // Disable brownout problems

// Select camera model
// #define CAMERA_MODEL_WROVER_KIT // Has PSRAM
//#define CAMERA_MODEL_ESP_EYE // Has PSRAM
//#define CAMERA_MODEL_M5STACK_PSRAM // Has PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE  // Has PSRAM
#define CAMERA_MODEL_AI_THINKER // Has PSRAM
//#define CAMERA_MODEL_TTGO_T_JOURNAL // No PSRAM

#include "esp_camera.h"
#include "camera_pins.h"
#include "downsample.h"

// #include <fb_gfx.h>

camera_fb_t * fb = NULL;
uint16_t *buffer;

//tflite stuff
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 110 * 1024;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external

void init_camera(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;//PIXFORMAT_GRAYSCALE;//PIXFORMAT_JPEG;//PIXFORMAT_RGB565;// 
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.frame_size = FRAMESIZE_QVGA;//FRAMESIZE_96X96;//FRAMESIZE_QVGA;//FRAMESIZE_96X96;//
  
  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if(psramFound()){
    // config.frame_size = FRAMESIZE_QVGA;//FRAMESIZE_96X96;//
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    // config.frame_size = FRAMESIZE_QVGA;//FRAMESIZE_96X96;//
    config.jpeg_quality = 12;
    config.fb_count = 2;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    delay(1000);
    ESP.restart();
  }
#if defined(CAMERA_MODEL_M5STACK_WIDE)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif
}


void setup() {
  
  esp_task_wdt_init(60, true);
  // put your setup code here, to run once:
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);//disable brownout detector

  Serial.begin(115200);
  init_camera();
  // buffer = (uint16_t *) malloc(240*320*2);

  dstImage = (uint16_t *) malloc(DST_WIDTH * DST_HEIGHT*2);
  delay(200);
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  if (tensor_arena == NULL) {
    //allocate memory for TensorArena on PSRAM
    tensor_arena = (uint8_t *) ps_malloc(kTensorArenaSize);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  
  tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
 
  /*
  static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();
  

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
      ////
  */
  
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);

}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i<2; i++){
  fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
    }
    if(fb){
      esp_camera_fb_return(fb);
      fb=NULL;
    }
  delay(1);
  }

  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
  }
  uint16_t * tmp = (uint16_t *) fb->buf;

  downsampleImage((uint16_t *) fb->buf, fb->width, fb->height);
  
  for (int y = 0; y < DST_HEIGHT; y++) {
    for (int x = 0; x < DST_WIDTH; x++) {
      tmp[y*(fb->width) + x] = (uint16_t) dstImage[y*DST_WIDTH +x];

    }
  }
    
  if(fb){
    esp_camera_fb_return(fb);
    fb = NULL;
  }

  int8_t * image_data = input->data.int8;
  //Serial.println(input->dims->size);

  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = ((uint16_t *) (dstImage))[i * kNumCols + j];

      // for inference
      uint8_t hb = pixel & 0xFF;
      uint8_t lb = pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);

      /**
      * Gamma corected rgb to greyscale formula: Y = 0.299R + 0.587G + 0.114B
      * for effiency we use some tricks on this + quantize to [-128, 127]
      */
      int8_t grey_pixel = ((305 * r + 600 * g + 119 * b) >> 10) - 128;

      image_data[i * kNumCols + j] = grey_pixel;
      
    }
  }

    

  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  

  TfLiteTensor* output = interpreter->output(0);
  

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  
  int8_t non_person_score = output->data.uint8[kNonPersonIndex];
  
  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float non_person_score_f =
      (non_person_score - output->params.zero_point) * output->params.scale;

  Serial.print("person score: "); Serial.println(person_score_f);
  Serial.print("non_person score: "); Serial.println(non_person_score_f);

}

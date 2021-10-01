// main file, for debug only
// sudo apt-get install freeglut3 freeglut3-dev libglew-dev

// g++ main.cpp  bilateral_slice_apply_array.cpp bilateral_slice_apply_cpu.cpp  -lGL  -lGLEW -lglut   -std=c++11

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "bilateral_slice_apply.h"
#include <cmath>
#include <chrono>
#include <random>


#define VALIDADE 0

void BilateralSliceApplyKernel_ORG(
    const float *grid, const float *guide, const float *input,
    const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float *out);



int execute_test_profile( int w, int h ) {
    //create some data
    int gw = 16;
    int gh = 16;
    int gd = 8;
    int g_channels = 12;
    int input_channels = 3;
    int output_channels = 3;

    hdrnet::bilateral_slice_apply::bilateral_slice_apply_data* data = hdrnet::bilateral_slice_apply::create_bilateral_slice_apply_data( w, h, gw, gh, gd, input_channels, output_channels, true );

    float* grid = new float[gw * gh * gd * g_channels];
    float* guide  = new float[ w * h   ];
    float* input = new float[ w * h * input_channels ];
    float* output = new float[w * h * output_channels];

    hdrnet::bilateral_slice_apply::init( data );
    hdrnet::bilateral_slice_apply::prepare( data );



    std::random_device rd;
    std::mt19937 e2( rd() );
    std::uniform_real_distribution<> dist( 0, 1.0 );
    for (int i = 0; i < gw * gh * gd * g_channels  ; i++) grid[i] = dist( e2 )* dist( e2 );
    for (int i = 0; i < w * h; i++) guide[i] = dist( e2 );
    for (int i = 0; i < w * h * input_channels; i++) input[i] = dist( e2 );
    for (int i = 0; i < w * h * output_channels; i++) output[i] = -1.0f;






    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    int frames_count = 1000;

    std::cout << "Start Computation" << std::endl;
    hdrnet::bilateral_slice_apply::eval( data, grid, guide, input, output );
    auto t1 = high_resolution_clock::now();
    for (int frame = 0; frame < frames_count; frame++) {
        hdrnet::bilateral_slice_apply::eval( data, grid, guide, input, output );
    }
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    duration<double, std::milli> ms_double = (t2 - t1);


    for (int i = 0; i < 200; i+=4) printf( "%8f %8f %8f %8f \n", output[i], output[i+1], output[i+2], output[i+3] );

    hdrnet::bilateral_slice_apply::free( data );
    std::cout << ms_double.count() / double( frames_count ) << "ms \n";
    return 0;
}


    int execute_test_hdrnet(int w, int h)
{
#include "validate_ax.hpp"

  if ((input_shape[1] != w) || (input_shape[0] != h))
  {
    std::cout << "Window Size must be " << input_shape[0] << " x " << input_shape[1] << std::endl;
    return -1;
  }

  int gw = grid_shape[1];
  int gh = grid_shape[0];
  int gd = grid_shape[2];
  int g_channels = grid_shape[3];
  int input_channels = input_shape[2];
  int output_channels = output_shape[2];

  hdrnet::bilateral_slice_apply::bilateral_slice_apply_data *data = hdrnet::bilateral_slice_apply::create_bilateral_slice_apply_data(w, h, gw, gh, gd, input_channels, output_channels, true);

  float *grid = grid_data;
  float *guide = guide_data;
  float *input = input_data;
  float *output = new float[w * h * output_channels];
  float *output_cpu = new float[w * h * output_channels];
  hdrnet::bilateral_slice_apply::init(data);
  hdrnet::bilateral_slice_apply::prepare(data);


 using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
int frames_count = 10000;

  std::cout << "Start Computation" << std::endl;
  hdrnet::bilateral_slice_apply::eval(data, grid, guide, input, output);
  auto t1 = high_resolution_clock::now();
  for(int frame = 0 ; frame < frames_count; frame++){
     hdrnet::bilateral_slice_apply::eval(data, grid, guide, input, output);
  }
  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1) ;
   duration<double, std::milli> ms_double = (t2 - t1);

  //BilateralSliceApplyKernel_ORG( grid,guide,input,h,w,gh,gw,gd,input_channels,output_channels, true, output_cpu );

          std::cout
      << "Done Computation" << std::endl;

  printf("input guide outout  OUTREAL \n");
  for (int xy = 0; xy < std::min(w * h,4000); xy++)
  {
    for (int k = 0; k < output_channels; k++)
    {
      int x = xy % w;
      int y = xy / w;
      float out_gl =   output[output_channels * xy + k];
      float out_cuda =   output_data[output_channels * xy + k];
      float out_cpu =   output_cpu[output_channels * xy + k];
      float err = fabs(out_gl - std::min(1.0f, out_cuda ));
      if (err > 0.01)
      {
        printf("(%3i %3i %1i)  gl:%f  cuda:%f  cpu:%f ", x, y, k,   out_gl, out_cuda , out_cpu);
        printf("   err(gl/cuda): %8f   pixels:%i\n", err , int(255*err) );
      }
    }
  }

  hdrnet::bilateral_slice_apply::free(data);


    std::cout << ms_double.count()/double(frames_count) << "ms \n";

  return 0;
}
int execute_test(int w, int h)
{
  int gw = 16;
  int gh = 16;
  int gd = 8;
  int g_channels = 12;
  int input_channels = 3;
  int output_channels = 3;
  hdrnet::bilateral_slice_apply::bilateral_slice_apply_data *data = hdrnet::bilateral_slice_apply::create_bilateral_slice_apply_data(w, h, gw, gh, gd, input_channels, output_channels, true);

  float *grid = new float[gw * gh * gd * (1 + input_channels) * output_channels];
  float *guide = new float[w * h];
  float *input = new float[w * h * input_channels];
  float *output = new float[w * h * output_channels];
  hdrnet::bilateral_slice_apply::init(data);
  hdrnet::bilateral_slice_apply::prepare(data);

  std::cout << "Start Computation" << std::endl;
  for (int j = 0; j < 10; j++)
    hdrnet::bilateral_slice_apply::eval(data, nullptr, guide, input, output);
  std::cout << "Done Computation" << std::endl;

  hdrnet::bilateral_slice_apply::free(data);

  return 0;
}

int main(int argc, char *argv[])
{

  int w,h;

  if ( VALIDADE  ){
  w = 60;
  h = 101;
  }
  else{
   w = 720;
   h = 1024;
  }
  glutInit(&argc, argv);
  // Global glInit
  glutInitWindowSize(w, h);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
  glutCreateWindow("renderer");
  glutHideWindow();

  GLenum err = glewInit();
  if (err != GLEW_OK)
  {
    std::cout << "Failed to initialize GLEW: " << err << std::endl;
    throw;
  }

  std::cout << "OpenGL context initialized." << std::endl;
  std::cout << "GL_VENDOR: " << glGetString(GL_VENDOR) << std::endl;
  std::cout << "GL_RENDERER: " << glGetString(GL_RENDERER) << std::endl;
  std::cout << "GL_VERSION: " << glGetString(GL_VERSION) << std::endl;


    if ( VALIDADE  ) { execute_test_hdrnet(w, h); }
    else{
       execute_test_profile(w,h);
    }

  glutExit();
  return 0;
}
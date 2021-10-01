
#include <algorithm>
 #include <cmath>


typedef long long int64 ;


#define OX_1D_KERNEL_LOOP(iii, nnn)   for (int iii = 0; iii < (nnn); iii += 1)



  float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x*x+eps);
}

  float d_diff_abs(float x) {
  float eps = 1e-8;
  return x/sqrt(x*x+eps);
}

  float weight_z(float x) {
  float abx = diff_abs(x);
  return std::max(1.0-abx, 0.0);
}

  float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {
    return 0.0f;
    // return abx;
  } else {
    return d_diff_abs(x);
  }
}
  void    BilateralSliceApplyKernel_ORG(

    const float* grid, const float* guide, const float* input,
    const int h, const int w,
    const int gh, const int gw, const int gd,
    const int input_chans, const int output_chans, const bool has_offset,
    float* out)
{
  // - Samples centered at 0.5.
  // - Repeating boundary conditions

  int grid_chans = input_chans*output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

 int64 nthreads = w * h *output_chans ;

  OX_1D_KERNEL_LOOP(idx, nthreads) {
    int out_c = idx % output_chans;
    int x = (idx / output_chans) % w;
    int y = (idx / (output_chans*w)) % h;
    int b = (idx / (output_chans*w*h));

    float gx = (x+0.5f)*gw/(1.0f*w);
    float gy = (y+0.5f)*gh/(1.0f*h);
    float gz = guide[x + w*(y + h*b)]*gd;

    int fx = static_cast<int>(floor(gx-0.5f));
    int fy = static_cast<int>(floor(gy-0.5f));
    int fz = static_cast<int>(floor(gz-0.5f));


    // Grid strides
    int sz = grid_chans;
    int sx = grid_chans*gd;
    int sy = grid_chans*gd*gw;
    int sb = grid_chans*gd*gw*gh;

    float value = 0.0f;
    for (int in_c = 0; in_c < coeff_stride; ++in_c) {
      float coeff_sample = 0.0f;
      for (int xx = fx; xx < fx+2; ++xx) {
        int x_ = std::max(std::min(xx, gw-1), 0);
        float wx = std::max(1.0-std::abs(xx+0.5-gx), 0.0);
        for (int yy = fy; yy < fy+2; ++yy)
        {
          int y_ = std::max(std::min(yy, gh-1), 0);
          float wy = std::max(1.0-std::abs(yy+0.5-gy), 0.0);
          for (int zz = fz; zz < fz+2; ++zz)
          {
            int z_ = std::max(std::min(zz, gd-1), 0);
            float wz = weight_z(zz+0.5-gz);
            //int grid_idx = (coeff_stride*out_c + in_c) + sz*z_ + sx*x_ + sy*y_ + sb*b;
            int grid_idx = (coeff_stride*out_c + in_c)+ sz*z_ + sx*x_ + sy*y_+ sb*b  ;
            coeff_sample += grid[grid_idx] *wx*wy*wz;
          }
        }

      } // Grid trilinear interpolation
      if(in_c < input_chans) {
        int input_idx = in_c + input_chans*(x + w*(y + h*b));
        value += coeff_sample* input[input_idx];
      } else { // Offset term
        value += coeff_sample;
      }
    }
     out[idx] = value;

  }
}
